# Copyright (c) OpenMMLab. All rights reserved.
import logging

import numpy as np
import torch
from mmcv.utils import print_log

_integer_types = (
    np.byte,
    np.ubyte,  # 8 bits
    np.short,
    np.ushort,  # 16 bits
    np.intc,
    np.uintc,  # 16 or 32 or 64 bits
    np.int_,
    np.uint,  # 32 or 64 bits
    np.longlong,
    np.ulonglong)  # 64 bits

_integer_ranges = {
    t: (np.iinfo(t).min, np.iinfo(t).max)
    for t in _integer_types
}

dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1)
}
dtype_range.update(_integer_ranges)


def dtype_limits(image, clip_negative=False):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    This function is adopted from skimage:
    https://github.com/scikit-image/scikit-image/blob/
    7e4840bd9439d1dfb6beaf549998452c99f97fdd/skimage/util/dtype.py#L35

    Args:
        image (ndarray): Input image.
        clip_negative (bool, optional): If True, clip the negative range
            (i.e. return 0 for min intensity) even if the image dtype allows
            negative values.

    Returns
        tuple: Lower and upper intensity limits.
    """
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


def adjust_gamma(image, gamma=1, gain=1):
    """Performs Gamma Correction on the input image.

    This function is adopted from skimage:
    https://github.com/scikit-image/scikit-image/blob/
    7e4840bd9439d1dfb6beaf549998452c99f97fdd/skimage/exposure/
    exposure.py#L439-L494

    Also known as Power Law Transform.
    This function transforms the input image pixelwise according to the
    equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.

    Args:
        image (ndarray): Input image.
        gamma (float, optional): Non negative real number. Defaults to 1.
        gain (float, optional): The constant multiplier. Defaults to 1.

    Returns:
        ndarray: Gamma corrected output image.
    """
    if np.any(image < 0):
        raise ValueError('Image Correction methods work correctly only on '
                         'images with non-negative values. Use '
                         'skimage.exposure.rescale_intensity.')

    dtype = image.dtype.type

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number.')

    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    out = ((image / scale)**gamma) * scale * gain
    return out.astype(dtype)


def random_choose_unknown(unknown, crop_size):
    """Randomly choose an unknown start (top-left) point for a given crop_size.

    Args:
        unknown (np.ndarray): The binary unknown mask.
        crop_size (tuple[int]): The given crop size.

    Returns:
        tuple[int]: The top-left point of the chosen bbox.
    """
    h, w = unknown.shape
    crop_h, crop_w = crop_size
    delta_h = center_h = crop_h // 2
    delta_w = center_w = crop_w // 2

    # mask out the validate area for selecting the cropping center
    mask = np.zeros_like(unknown)
    mask[delta_h:h - delta_h, delta_w:w - delta_w] = 1
    if np.any(unknown & mask):
        center_h_list, center_w_list = np.where(unknown & mask)
    elif np.any(unknown):
        center_h_list, center_w_list = np.where(unknown)
    else:
        print_log('No unknown pixels found!', level=logging.WARNING)
        center_h_list = [center_h]
        center_w_list = [center_w]
    num_unknowns = len(center_h_list)
    rand_ind = np.random.randint(num_unknowns)
    center_h = center_h_list[rand_ind]
    center_w = center_w_list[rand_ind]

    # make sure the top-left point is valid
    top = np.clip(center_h - delta_h, 0, h - crop_h)
    left = np.clip(center_w - delta_w, 0, w - crop_w)

    return top, left


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers.

    Args:
        shape (tuple): shape of image.
        ranges (tuple): range of coordinate value. Default: None.
        flatten (bool): flatten to (n, 2) or Not. Default: True.

    return:
        coord (Tensor): coordinates.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    coord = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        coord = coord.view(-1, coord.shape[-1])
    return coord
