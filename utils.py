import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import yaml
from tqdm import tqdm

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edist

import torch
from hdf5storage import loadmat  # load matrices

from network_utils import scale_tensor, get_masks

from features import *


def get_device(DEVICE: str=None) -> torch.device:
    """
    Get device to use for PyTorch.
    :param DEVICE: "C
    :return: "cuda" if cuda is available, else "cpu"
    """
    # Check if specified device is valid
    assert DEVICE is None or DEVICE.lower() == "cpu" or DEVICE.lower() == "cuda", "Invalid device specified."

    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(DEVICE)


def get_model_parameters(model: torch.nn.Module):
    """
    Get model parameters.
    :param model:
    :return:
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


_MIN = 0
_MAX = 1000

"""
model utilities
"""


def load_hparams(yaml_loc):
    with open(yaml_loc, 'r') as stream:
        # return yaml.safe_load(stream)
        return yaml.load(stream, Loader=yaml.Loader)


def create_folder(params):
    # let's save the dict too

    import datetime
    from shutil import copyfile, move, SameFileError

    model_name = params['net_name']

    def rename_file(file_name):
        try:
            move(f'savedModels/{model_name}/{file_name}',
                 f'savedModels/{model_name}/' +
                 datetime.datetime.today().strftime("%d_%H_%M_") +
                 f'{file_name}')
        except SameFileError:
            pass
        except:
            print(f'Seems like {file_name} is not here :(')

    try:
        os.mkdir(f'savedModels/{model_name}')
        os.mkdir(f'savedModels/{model_name}/plots')
        print(f"Directory {model_name} created")

    except FileExistsError:
        print(f"Directory {model_name} already exists")
        for file_n in ['train.py', 'net_dict.json',
                       'results_dict.json', 'losses.png']:
            rename_file(file_n)

    copyfile('train.py', f'savedModels/{model_name}/train.py')


"""
data loading
"""


def data4test(data, net_dict, half=False):
    data = movedata(data, net_dict['device'])
    if half:
        data = tohalf(data)
    return data[1], data[-1][0], data[-1][1]#int(data[0].item()), data[1], data[-1][0], data[-1][1]


#          #samplenum,      masks,         xs,                ys


def sortdata(data, net_dict):
    # returns stacked features
    num_xs = len(net_dict['x_array'])
    return [[torch.cat(feats) for feats in zip(*data[:num_xs])], *data[num_xs:]]


def tohalf(data):
    return [elem.half() if type(elem) == torch.Tensor else
            tohalf(elem) if type(elem) == list else elem for elem in data]


def movedata(data, device):
    return [elem.to(device) if type(elem) == torch.Tensor else
            movedata(elem, device) if type(elem) == list else elem for elem in data]


def check_inputs(input_dict, phase):
    assert len(input_dict[f'{phase}_sample']) == \
           len(input_dict[f'{phase}_subsample']) == \
           len(input_dict[f'{phase}_size']) == \
           len(input_dict[f'{phase}_pressure'])


def return_fields(net_dict, phase):
    samples = net_dict[f'{phase}_sample']
    try:
        subsamples = [format(x, '02') for x in net_dict[f'{phase}_subsample']]
    except ValueError:  # if the string is correctly formatted
        subsamples = net_dict[f'{phase}_subsample']
    sizes = net_dict[f'{phase}_size']
    pressures = net_dict[f'{phase}_pressure']

    return samples, subsamples, sizes, pressures


def get_dataloader(net_dict, phases):
    from torch.utils.data import DataLoader

    if isinstance(phases, str) == True:  # when only testing is needed
        phases = [phases]

    """The dataloader will have the following structure per sample:
        - DL[0]: sample number (int)
        - DL[1]: list of masks of size = num of scales, where the last one is binary
        - DL[2]: list of inputs/outputs
            i.e. DL[2][0][-1] should be the largest input image, conversely
            i.e. DL[2][-1][-1] should be the largest output (y)
      """

    dataloader = {}
    for phase in phases:
        check_inputs(net_dict, phase)
        samples, subsamples, sizes, pressures = return_fields(net_dict, phase)
        data = []

        for num, name_tuple in enumerate(zip(samples, subsamples, sizes, pressures)):
            if num != 0:
                if num < _MIN or num > _MAX:  # memory limit
                    continue
            sample_name = "".join([str(e) + '_' for e in name_tuple])
            data_tmp = get_sample(net_dict, sample_name)
            if len(net_dict['x_array']) > 1:
                data_tmp = sortdata(data_tmp, net_dict)  # concat feats
            #                         [x][fine][feat0]
            masks = get_masks(data_tmp[0][-1][0][None, None,], net_dict['num_scales'])
            data.append((num,) + (masks,) + (data_tmp,))
        dataloader[phase] = data
                    # DataLoader(data, batch_size=net_dict['batch_size'],
                    #                    shuffle=(phase == 'train'),
                    #                    # shuffle = False
                    #                    pin_memory=True,
                    #                    num_workers=net_dict['num_workers'])
    return dataloader


def get_sample(net_dict, sample_name):
    tmp_dict = [load_samples(feat, sample_name, net_dict, xform=xform)
                for feat, xform in zip(
            [*net_dict['x_array'], *net_dict['y_array']],
            [*net_dict['x_xform'], *net_dict['y_xform']])]
    return [get_downscaled_list(im_array, net_dict) for im_array in tmp_dict]


def get_downscaled_list(x, net_dict):
    """
    X : 3D np array
    returns a list with the desired number of coarse-grained tensors
    """
    x = torch.Tensor(add_dims(x, 1))
    ds_x = []
    ds_x.append(x)
    for i in range(net_dict['num_scales'] - 1):
        ds_x.append(scale_tensor(ds_x[-1], scale_factor=1 / 2, mode='nearest'))
    return ds_x[::-1]  # returns the reversed list (small images first)


"""
data stats
"""


def get_sum_stats(feat, net_dict, phase):
    """
    Returns the summary stats of a feature over multiple samples
    """

    check_inputs(net_dict, phase)
    samples, subsamples, sizes, pressures = return_fields(net_dict, phase)
    sample_list = []
    for name_tuple in zip(samples, subsamples, sizes, pressures):
        sample_name = "".join([str(e) + '_' for e in name_tuple])
        sample_list.append(load_samples(feat, sample_name, net_dict))

    return sum_stats(sample_list, remove_zeros=True)


def load_samples(feat, sample_name, net_dict, xform=None):
    """
    feat: either mpf, edist or uz
    sample_name: sample num
    xform: data transform to perform
    """

    data_dict = {'bin': {'path': f'{net_dict["data_loc"]}/bin', 'dkey': 'bin', 'ext': 'bin'},
                 'phi': {'path': f'{net_dict["data_loc"]}/elec', 'dkey': 'phi', 'ext': 'elec'},
                 'Iz': {'path': f'{net_dict["data_loc"]}/elec', 'dkey': 'Iz', 'ext': 'elec'}}

    path = data_dict.get(feat, data_dict.get('bin'))['path']

    sample = loadmat(f'{path}/{sample_name}{data_dict.get(feat, data_dict.get("bin"))["ext"]}.mat')[
        data_dict.get(feat, data_dict.get("bin"))["dkey"]]

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

    if not (feat == 'phi' or feat == 'Iz'):
        assert feat in feature_dict, NotImplemented(
            "Selected feature has not been implemented. Please add it to features.py")
        # Inverted binary image
        sample_inv = -1 * sample + 1
        sample = feature_dict[feat](sample_inv)

    if xform is not None:
        # net_dict[f'{feat}_stats'] = sum_stats(sample)
        sumstats_dict = net_dict[f'{feat}_stats']  # summary statistics
        sample_copy = np.copy(sample)
        sample = data_xform(xform)(sample, sumstats_dict)  # transforms data
        sample[sample_copy == 0] = 0
        del sample_copy

    sample[~np.isfinite(sample)] = 0
    sample = zero_bounds(sample)

    return sample


def sum_stats(x, remove_zeros=False):
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

def get_coarsened_list(x, scales):
    """
    X : 3D np array
    returns a list with the desired number of coarse-grained tensors
    """

    # converts to tensor and adds channel and batch dim
    x = torch.Tensor(add_dims(x, 1))

    ds_x = []
    ds_x.append(x)

    for i in range(scales - 1):
        ds_x.append(scale_tensor(ds_x[-1], scale_factor=1 / 2))
    return ds_x[::-1]  # returns the reversed list (small images first)


def zero_bounds(image):
    """
    Make all boundary faces zero. This is useful because Dirichlet BCs are enforced in these voxels, and therefore, do not need to be trained.
    image: 3D ndarray
    returns 3D ndarray copy image with boundary faces set equal to zero.
    """

    zero_bound = np.zeros_like(image)
    zero_bound[1:-1, 1:-1, 1:-1] = image[1:-1, 1:-1, 1:-1]
    return zero_bound


"""
Tensor operations
"""

def changepres(x, ttype=None):
    if ttype == 'f32':
        return x.float()
    elif ttype == 'f16':
        return x.half()


def add_dims(x, num_dims):
    for dims in range(num_dims):
        x = x[np.newaxis]
    return x


"""
Data transforms
"""


def inv_xform(xform):
    if xform == 'minMax':
        return inv_minMax
    elif xform == 'div_maxabs':
        return inv_div_maxabs
    elif xform == 'div_scalar':
        return inv_div_scalar
    else:
        raise NotImplementedError


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



