from network_utils import get_masks, get_downscaled_list
import torch
import numpy as np
import tifffile
from input_features import *
from hdf5storage import loadmat



def check_inputs(input_dict, phase):
    assert len(input_dict[f'{phase}_client']) == \
           len(input_dict[f'{phase}_sample']) == \
           len(input_dict[f'{phase}_size'])


def get_fields(net_dict, phase):
    """
    Get the fields of data files
    Args:
        net_dict:
        phase:

    Returns:

    """
    clients = net_dict[f'{phase}_client'] * len(net_dict[f'{phase}_sample'])
    try:
        # subsamples = [format(x, '02') for x in net_dict[f'{phase}_sample']]
        # 3 digits with leading zeros
        samples = [f"{x:03d}" for x in net_dict[f'{phase}_sample']]
    except ValueError:  # if the string is correctly formatted
        samples = net_dict[f'{phase}_sample']

    # Assume all sizes are equal
    sizes = net_dict[f'{phase}_size'] * len(net_dict[f'{phase}_sample'])

    return clients, samples, sizes


def get_sample(net_dict, sample_name):
    tmp_dict = [load_samples(feat, sample_name, net_dict, xform=xform)
                for feat, xform in zip(
            [*net_dict['x_array'], *net_dict['y_array']],
            [*net_dict['x_xform'], *net_dict['y_xform']])]
    return [get_downscaled_list(im_array, net_dict) for im_array in tmp_dict]


def load_samples(feat, sample_name, net_dict, xform=None):
    """
    feat: either mpf, edist or uz
    sample_name: sample num
    xform: data transform to perform
    """
    # TODO: Add data location to hydra config file
    data_dict = {'bin': {'path': f'{net_dict["data_loc"]}/bin', 'dkey': 'bin', 'ext': 'bin'},
                 'phi': {'path': f'{net_dict["data_loc"]}/elec', 'dkey': 'phi', 'ext': 'elec'},
                 'Iz': {'path': f'{net_dict["data_loc"]}/elec', 'dkey': 'Iz', 'ext': 'elec'},
                 'vel': {'path': f'{net_dict["data_loc"]}/vel', 'dkey': 'uz', 'ext': 'uz'}}

    path = data_dict.get(feat, data_dict.get('bin'))['path']

    # TODO: load data using tifffile instead
    # sample = loadmat(f'{path}/{sample_name}{data_dict.get(feat, data_dict.get("bin"))["ext"]}.mat')[
    #     data_dict.get(feat, data_dict.get("bin"))["dkey"]]


    sample = tifffile.imread(f'{path}/{sample_name}{data_dict.get(feat, data_dict.get("bin"))["ext"]}.tif')

    # if feat == 'bin':
    #    sample = -1*sample + 1

    # if feat == 'phi':
    #    linear = linear_trend(sample)
    #    sample = sample - linear

    feature_dict = {'bin': binary_img,
                    'slicewise_edt': slicewise_edt,
                    'edt': edt,
                    'sdt': sdt,
                    'slicewise_sdt': slicewise_sdt,
                    'slicewise_mis': slicewise_mis,
                    'chords_x': chords_x,
                    'chords_y': chords_y,
                    'ellipse_area': ellipse_area,
                    'tofl': tofl,
                    'tofr': tofr,
                    'cf': cf,
                    'porosity': slicewise_porosity,
                    'linear': linear_trend}

    if not (feat == 'phi' or feat == 'Iz' or feat == 'vel'):
        assert feat in feature_dict, NotImplemented(
            "Selected feature has not been implemented. Please add it to features.py")
        # Inverted binary image
        sample_inv = -1 * sample + 1
        sample = feature_dict[feat](sample_inv)

    if xform is not None:
        net_dict[f'{feat}_stats'] = sum_stats(sample)
        sumstats_dict = net_dict[f'{feat}_stats']  # summary statistics
        sample_copy = np.copy(sample)
        sample = data_xform(xform)(sample, sumstats_dict)  # transforms data
        sample[sample_copy == 0] = 0
        del sample_copy

    sample[~np.isfinite(sample)] = 0
    # sample = zero_bounds(sample)

    return sample

def zero_bounds(image):
    """
    Make all boundary faces zero. This is useful because Dirichlet BCs are enforced in these voxels, and therefore, do not need to be trained.
    image: 3D ndarray
    returns 3D ndarray copy image with boundary faces set equal to zero.
    """

    zero_bound = np.zeros_like(image)
    zero_bound[1:-1, 1:-1, 1:-1] = image[1:-1, 1:-1, 1:-1]
    return zero_bound

def sortdata(data, net_dict):
    # returns stacked features
    num_xs = len(net_dict['x_array'])
    return [[torch.cat(feats) for feats in zip(*data[:num_xs])], *data[num_xs:]]



def get_sum_stats(feat, net_dict, phase):
    """
    Returns the summary stats of a feature over multiple samples
    """

    check_inputs(net_dict, phase)
    samples, subsamples, sizes, pressures = get_fields(net_dict, phase)
    sample_list = []
    for name_tuple in zip(samples, subsamples, sizes, pressures):
        sample_name = "".join([str(e) + '_' for e in name_tuple])
        sample_list.append(load_samples(feat, sample_name, net_dict))

    return sum_stats(sample_list, remove_zeros=True)

def sum_stats(x, remove_zeros=True):
    """
    Returns a dictionary with the summary stats of x
    """

    t_dic = {}  # dict for data transforms

    if remove_zeros == True:
        x = np.concatenate([xi[xi != 0] for xi in x])

    t_dic['min'] = np.min(x)
    t_dic['range'] = np.ptp(x)
    t_dic['max'] = np.max(x)
    t_dic['max_abs'] = np.max(np.abs(x))
    t_dic['std'] = np.std(x)
    t_dic['mean'] = np.mean(x)
    return t_dic


def all_sum_stats(net_dict):
    """
    Calculates the summary statistics per feature
    """
    for feat in [*net_dict['x_array'], *net_dict['y_array']]:
        # get the summary statistics
        net_dict[f'{feat}_stats'] = get_sum_stats(feat, net_dict, 'train')
    return net_dict


def data_xform(xform):
    if xform == 'minMax':
        return minMax
    elif xform == 'div_maxabs':
        return div_maxabs
    elif xform == 'div_mean':
        return div_mean
    elif xform == 'div_scalar':
        return div_scalar
    elif xform == 'std':
        return std_t
    else:
        raise NotImplementedError


def minMax(x, t_dict):
    if type(x) == list:
        return [minMax(x0, t_dict) for x0 in x]
    else:
        return (np.copy(x) - t_dict['min']) / t_dict['range']


def div_maxabs(x, t_dict):
    if type(x) == list:
        return [div_maxabs(x0, t_dict) for x0 in x]
    else:
        return np.copy(x) / t_dict['max_abs']


def div_scalar(x, t_dict):
    if type(x) == list:
        return [div_scalar(x0, t_dict) for x0 in x]
    else:
        # return  np.copy(x)/1e-9
        return np.copy(x) / t_dict['scalar']


def div_mean(x, t_dict):
    if type(x) == list:
        return [div_mean(x0, t_dict) for x0 in x]
    else:
        return np.copy(x) / t_dict['mean']


def std_t(x, t_dict):
    if type(x) == list:
        return [std_t(x0, t_dict) for x0 in x]
    else:
        return (np.copy(x) - t_dict['mean']) / t_dict['std']


def inv_minMax(xt, t_dict):
    return t_dict['range'] * (xt) + t_dict['min']


def inv_div_maxabs(xt, t_dict):
    return t_dict['max_abs'] * (xt)


def inv_div_scalar(xt, t_dict):
    # return xt*1e-9
    return xt * t_dict['scalar']
