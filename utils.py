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



