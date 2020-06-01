import numpy as np


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
    center_h_list, center_w_list = np.where(unknown & mask)
    num_unknowns = len(center_h_list)
    if num_unknowns > 0:
        rand_ind = np.random.randint(num_unknowns)
        center_h = center_h_list[rand_ind]
        center_w = center_w_list[rand_ind]

    # if crop_size has odd number, make sure the top-left point is valid
    top = min(h - crop_h, center_h - delta_h)
    left = min(w - crop_w, center_w - delta_w)

    return top, left
