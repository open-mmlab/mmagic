# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn.functional as F


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_args: dict = dict()):
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images.

    If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.
    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_args (dict): The padding args.
    Returns:
        batch_tensor (Tensor): The 4D-tensor or 5D-tensor.
            Tensor.dim == tensor_list[0].dim + 1
        padded_sizes (Tensor): The padded sizes of each tensor.
    """

    assert isinstance(tensor_list, list), ('Expected input type to be list, '
                                           f'but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The dim of channel and frame index should not be padded.
    padded_sizes[:, :-2] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list), padded_sizes

    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), **pad_args))
    return torch.stack(batch_tensor), padded_sizes


def split_batch(batch_tensor: torch.Tensor, padded_sizes: torch.Tensor):
    """reverse operation of ``stack_batch``.

    Args:
        batch_tensor (Tensor): The 4D-tensor or 5D-tensor.
            Tensor.dim == tensor_list[0].dim + 1
        padded_sizes (Tensor): The padded sizes of each tensor.
    Returns:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
    """

    free_tensors = []
    for idx in range(batch_tensor.shape[0]):
        tensor = batch_tensor[idx]
        if padded_sizes.sum() > 0:
            padded_h, padded_w = padded_sizes[idx][-2:]
            padded_h = int(padded_h)
            padded_w = int(padded_w)
            h, w = tensor.shape[-2:]
            tensor = tensor[..., :h - padded_h, :w - padded_w]
        free_tensors.append(tensor)

    return free_tensors
