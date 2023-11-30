from torch.utils.data import DataLoader, random_split
import torch
from typing import List

from dataloading_utils import *
import yaml
import os


def load_data(data_input, path_to_data="./training_data", phases=('train', 'val', 'test')):
    """
    Load data listed in input yaml file
    Args:
        data_input: Name of yaml file to load data from
        path_to_data: Path to parent directory containing data
        phases: List of phases to load data: options are 'train', 'val', 'test'
    Returns:
        training dataloader, validation dataloader
    """
    with open(os.path.join("./data_input_files", data_input), 'r') as stream:
        input_samples = yaml.load(stream, Loader=yaml.Loader)

    assert all(phase.lower() in ['train', 'val', 'test'] for phase in phases), \
        "Phase must be one of 'train', 'val', or 'test'"

    input_samples['x_xform'] = [None if xform == "None" else xform for xform in input_samples['x_xform']]
    input_samples['y_xform'] = [None if xform == "None" else xform for xform in input_samples['y_xform']]
    input_samples['data_loc'] = path_to_data

    data_sets = []
    for phase in phases:
        data_sets.append(get_dataloader(input_samples, [phase])[phase])

    return data_sets


def get_dataloader(net_dict, phases):

    if isinstance(phases, str):  # when only testing is needed
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
        samples, subsamples, sizes, pressures = get_fields(net_dict, phase)
        data = []

        for num, name_tuple in enumerate(zip(samples, subsamples, sizes, pressures)):
            if num != 0:
                if num < 0 or num > 1000:  # memory limit
                    continue
            sample_name = "".join([str(e) + '_' for e in name_tuple])
            data_tmp = get_sample(net_dict, sample_name)
            if len(net_dict['x_array']) > 1:
                data_tmp = sortdata(data_tmp, net_dict)  # concat feats
            # TODO: change num_scales to be read from config file
            num_scales = 4
            masks = get_masks(data_tmp[0][-1][0][None, None], num_scales)
            data.append((num,) + (masks,) + (data_tmp,))
        dataloader[phase] = DataLoader(data, batch_size=1,
                                       shuffle=(phase=='train'),
                                       pin_memory=True,
                                       num_workers=int(os.cpu_count()//2))
    return dataloader
