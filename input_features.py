import numpy as np
import porespy as ps
from scipy.ndimage.morphology import distance_transform_edt as edist
from typing import Tuple


def binary_img(image) -> np.ndarray:
    """
    Returns binary image where 1 is the pore space and 0 is the grain space
    """
    return image


def edt(image) -> np.ndarray:
    """
    Returns Euclidean distance transform map of the entire 3D image
    """
    return edist(image)


def sdt(image) -> np.ndarray:
    """
    Returns Signed distance transform map of the input image
    """

    grain = -1 * edt(image)
    sample = -1 * image + 1
    pore = edt(sample)
    sample = grain + pore

    return sample


def slicewise_edt(image) -> np.ndarray:
    """
    Returns Euclidean distance transform map of each slice individually
    """
    slice_edt = np.zeros_like(image, dtype=np.float32)
    for s in range(image.shape[2]):
        slice_edt[:, :, s] = edt(image[:, :, s])

    return slice_edt


def slicewise_sdt(image) -> np.ndarray:
    """
    Returns signed distance transform map of each slice individually
    """
    slice_sdt = np.zeros_like(image, dtype=np.float32)
    for s in range(image.shape[2]):
        slice_sdt[:, :, s] = sdt(image[:, :, s])

    return slice_sdt


def slicewise_mis(image, **kwargs) -> np.ndarray:
    """
    A function that calculates the slice-wise maximum inscribed sphere (maximum inscribed disk)
    """
    # TODO why do we pad this?
    input_image = np.pad(array=image.copy(), pad_width=((0, 0), (0, 0), (0, 1)), constant_values=1)

    # Calculate slice-wise local thickness from PoreSpy
    thickness = np.zeros_like(input_image)
    for img_slice in range(image.shape[2] + 1):
        thickness[:, :, img_slice] = ps.filters.local_thickness(input_image[:, :, img_slice], sizes=40, mode='hybrid',
                                                                divs=4)

    return thickness


def chords(image, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A function that calculates the ellipse area based on the chord lengths in slices orthogonal to direction of flow
    Assumes img = 1 for pore space, 0 for grain space
    Returns length of chords in x, y, and ellipse areas
    """
    ellipse_area = np.zeros_like(image, dtype=np.float32)
    sz_x = np.zeros_like(image, dtype=np.float32)
    sz_y = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[2]):
        # Calculate the chords in x and y for each slice in z
        chords_x = ps.filters.apply_chords(im=image[:, :, i], spacing=0, trim_edges=False, axis=0)
        chords_y = ps.filters.apply_chords(im=image[:, :, i], spacing=0, trim_edges=False, axis=1)

        # Get chord lengths
        sz_x[:, :, i] = ps.filters.region_size(chords_x)
        sz_y[:, :, i] = ps.filters.region_size(chords_y)

        # Calculate ellipse area from chords
        ellipse_area[:, :, i] = np.pi / 4 * sz_x[:, :, i] * sz_y[:, :, i]

    return sz_x, sz_y, ellipse_area


def chords_x(image):
    """
    Helper function to extract chords in x direction
    """
    return chords(image)[0]


def chords_y(image):
    """
    Helper function to extract chords in y direction
    """
    return chords(image)[1]


def ellipse_area(image):
    """
    Helper function to ellipse area
    """
    return chords(image)[2]


def tof(image, boundary: str = 'l', detrend: bool = False) -> np.ndarray:
    """
    Get time of flight map (solution to Eikonal equation) from specified boundary (inlet or outlet)
    Assumes img = 1 in pore space, 0 for grain space
    Assumes flow is in z direction (orthogonal to axis 2)
    """
    inlet = np.zeros_like(image)
    if boundary[0].lower() == 'l':
        inlet[:, :, 0] = 1.
    elif boundary[0].lower() == 'r':
        inlet[:, :, -1] = 1.
    else:
        raise KeyError("Invalid inlet boundary")

    # Calculate ToF of input image
    tof_map = ps.tools.marching_map(image, inlet)

    if detrend:
        tmp = np.ones_like(image)
        trend = ps.tools.marching_map(tmp, inlet)
        trend *= image  # Mask trended image to obey solid matrix
        tof_map -= trend

    return tof_map


def tofl(image):
    """
    Helper function to get time of flight from the inlet
    """
    return tof(image, boundary='l', detrend=False)


def tofr(image):
    """
    Helper function to get time of flight from the outlet
    """
    return tof(image, boundary='r', detrend=False)


def constriction_factor(thickness_map: np.ndarray, power: float = None) -> np.ndarray:
    """
    A function that calculates the slice-wise constriction factor from the input thickness map.
    Constriction factor defined as thickness[x, y, z] / thickness[x, y, z+1]
    Padded with reflected values at outlet
    """
    thickness_map = np.pad(thickness_map.copy(), ((0, 0), (0, 0), (0, 1)), 'reflect')

    if power is not None:
        thickness_map = np.power(thickness_map, power)

    constriction_map = np.divide(thickness_map[:, :, :-1], thickness_map[:, :, 1:])

    # Change constriction to 0 if directly preceding solid matrix (is this the right thing to do?)
    constriction_map[np.isinf(constriction_map)] = 0

    # Change constriction to 0 if directly behind solid matrix
    constriction_map[np.isnan(constriction_map)] = 0

    return constriction_map


def cf(image):
    """
    Helper function to get constriction factor based on ellipse area
    """
    thickness_map = ellipse_area(image)
    return constriction_factor(thickness_map)


def slicewise_porosity(image):
    """
    Replace all pore values with the porosity of the slice
    """

    porosity_map = np.copy(image).astype(np.float32)

    for s in range(image.shape[2]):
        slice_porosity = np.count_nonzero(image[:, :, s]) / (image.shape[0] * image.shape[1])
        porosity_map[:, :, s][porosity_map[:, :, s] > 0] = slice_porosity

    return porosity_map


def linear_trend(image, inlet=2, outlet=1):
    """
    Create a linear trend through the pore space based on Digital Rock Suite BCs
    """
    # Make mask
    binary_mask = (image != 0).astype(np.uint8)
    trend = np.linspace(inlet - 1 / image.shape[0], outlet - 1 / image.shape[0], image.shape[0])
    trend = np.broadcast_to(trend, image.shape)

    # Mask grains
    trended_img = trend * binary_mask

    return trended_img