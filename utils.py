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
from flwr.common.parameter import ndarrays_to_parameters


def get_device(DEVICE: str=None) -> torch.device:
    """
    Get device to use for PyTorch.
    :param DEVICE: "C
    :return: "cuda" if cuda is available, else "cpu"
    """
    # Check if specified device is valid
    assert DEVICE is None or DEVICE.lower() == "cpu" or DEVICE.lower() == "cuda", "Invalid device specified."

    if DEVICE is None:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    return DEVICE#torch.device(DEVICE)


def get_model_parameters(model: torch.nn.Module):
    """
    Get model parameters.
    :param model:
    :return:
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def write_yaml(data_list: list, phases: list, filename: str):
    """
    Convert data from list of lists to yaml file.
    :param data_list: List of lists for each phase. Must match length of phases and the order of phases.
    :param phases: List of phases, i.e. ["train", "val"]
    """

    assert len(data_list) == len(phases), "Length of data_list must match length of phases."

    data_dict = {}
    for i, phase in enumerate(phases):
        sample, subsample, size, pressure = [], [], [], []
        for data in data_list[i]:
            splits = data.split("_")
            sample.append(splits[0])
            subsample.append(format(int(splits[1]), '02'))
            size.append(splits[2])
            pressure.append(splits[3])

        data_dict[f"{phase}_sample"] = sample
        data_dict[f"{phase}_subsample"] = subsample
        data_dict[f"{phase}_size"] = size
        data_dict[f"{phase}_pressure"] = pressure

    with open(f"./data_input_files/{filename}", "w") as f:
        yaml.dump(data_dict, f)

def model_to_parameters(model):
    """ Note that the model is already instantiated when passing it here

    This happens because we call this utility function when instantiating the parent
    object (i.e. the FedAdam strategy in this example).
    """
    ndarrays = get_model_parameters(model)
    parameters = ndarrays_to_parameters(ndarrays)

    return parameters
