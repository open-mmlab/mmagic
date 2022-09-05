# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Union

import torch
from torch import Tensor


def noise_sample_fn(noise: Union[Tensor, Callable, None] = None,
                    *,
                    num_batches: int = 1,
                    noise_size: Union[int, Sequence[int], None] = None,
                    device: Optional[str] = None) -> Tensor:

    if isinstance(noise, torch.Tensor):
        noise_batch = noise
        # unsqueeze if need
        if noise_batch.ndim == 1:
            noise_batch = noise_batch[None, ...]
    else:
        # generate noise automatically, prepare and check `num_batches` and
        # `noise_size`
        # num_batches = 1 if num_batches is None else num_batches
        assert num_batches > 0, (
            'If you want to generate noise automatically, \'num_batches\' '
            'must larger than 0.')
        assert noise_size is not None, (
            'If you want to generate noise automatically, \'noise_size\' '
            'must not be None.')
        noise_size = [noise_size] if isinstance(noise_size, int) \
            else noise_size

        if callable(noise):
            # receive a noise generator and sample noise.
            noise_generator = noise
            noise_batch = noise_generator((num_batches, *noise_size))
        else:
            # otherwise, we will adopt default noise sampler.
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, *noise_size))

    # Check the shape if `noise_size` is passed. Ignore `num_batches` here
    # because `num_batches` has default value.
    if noise_size is not None:
        if isinstance(noise_size, int):
            noise_size = [noise_size]
        assert list(noise_batch.shape[1:]) == noise_size, (
            'Size of the input noise is inconsistency with \'noise_size\'\'. '
            f'Receive \'{noise_batch.shape[1:]}\' and \'{noise_size}\' '
            'respectively.')

    if device is not None:
        return noise_batch.to(device)
    return noise_batch


def label_sample_fn(label: Union[Tensor, Callable, List[int], None] = None,
                    *,
                    num_batches: int = 1,
                    num_classes: Optional[int] = None,
                    device: Optional[str] = None) -> Union[Tensor, None]:

    if num_classes is None or num_classes <= 0:
        return None
    if isinstance(label, Tensor):
        label_batch = label
    elif isinstance(label, list):
        label_batch = torch.stack(label, dim=0)
    else:
        # generate label_batch manually, prepare and check `num_batches`
        assert num_batches > 0, (
            'If you want to generate label automatically, \'num_batches\' '
            'must larger than 0.')

        if callable(label):
            # receive a noise generator and sample noise.
            label_generator = label
            label_batch = label_generator(num_batches)
        else:
            # otherwise, we will adopt default label sampler.
            label_batch = torch.randint(0, num_classes, (num_batches, ))

    # Check the noise if `num_classes` is passed. Ignore `num_batches` because
    # `num_batches` has default value.
    if num_classes is not None:
        invalid_index = torch.logical_or(label_batch < 0,
                                         label_batch >= num_classes)
        assert not (invalid_index).any(), (
            f'Labels in label_batch must be in range [0, num_classes - 1] '
            f'([0, {num_classes}-1]). But found {label_batch[invalid_index]} '
            'are out of range.')

    if device is not None:
        return label_batch.to(device)
    return label_batch
