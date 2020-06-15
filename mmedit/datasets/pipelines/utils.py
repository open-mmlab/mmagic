import logging

import numpy as np
from mmcv.utils import print_log


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
