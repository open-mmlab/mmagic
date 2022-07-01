# Copyright (c) OpenMMLab. All rights reserved.

import torch


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.

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
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grids = torch.meshgrid(*coord_seqs, indexing='ij')
    else:
        grids = torch.meshgrid(*coord_seqs)
    coord = torch.stack(grids, dim=-1)
    if flatten:
        coord = coord.view(-1, coord.shape[-1])
    return coord
