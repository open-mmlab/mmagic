# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine import is_list_of
from torch import Tensor


def noise_sample_fn(noise: Union[Tensor, Callable, None] = None,
                    *,
                    num_batches: int = 1,
                    noise_size: Union[int, Sequence[int], None] = None,
                    device: Optional[str] = None) -> Tensor:
    """Sample noise with respect to the given `num_batches`, `noise_size` and
    `device`.

    Args:
        noise (torch.Tensor | callable | None): You can directly give a
            batch of noise through a ``torch.Tensor`` or offer a callable
            function to sample a batch of noise data. Otherwise, the ``None``
            indicates to use the default noise sampler. Defaults to None.
        num_batches (int, optional): The number of batch size. Defaults to 1.
        noise_size (Union[int, Sequence[int], None], optional): The size of
            random noise. Defaults to None.
        device (Optional[str], optional): The target device of the random
            noise. Defaults to None.

    Returns:
        Tensor: Sampled random noise.
    """
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
        assert list(noise_batch.shape[1:]) == list(noise_size), (
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
    """Sample random label with respect to `num_batches`, `num_classes` and
    `device`.

    Args:
        label (Union[Tensor, Callable, List[int], None], optional): You can
            directly give a batch of label through a ``torch.Tensor`` or offer
            a callable function to sample a batch of label data. Otherwise, the
            ``None`` indicates to use the default label sampler.
            Defaults to None.
        num_batches (int, optional): The number of batch size. Defaults to 1.
        num_classes (Optional[int], optional): The number of classes. Defaults
            to None.
        device (Optional[str], optional): The target device of the label.
            Defaults to None.

    Returns:
        Union[Tensor, None]: Sampled random label.
    """
    # label is not passed and do not have `num_classes` to sample label
    if (num_classes is None or num_classes <= 0) and (label is None):
        return None

    if isinstance(label, Tensor):
        label_batch = label
    elif isinstance(label, np.ndarray):
        label_batch = torch.from_numpy(label).long()
    elif isinstance(label, list):
        if is_list_of(label, (int, np.ndarray)):
            label = [torch.LongTensor([lab]).squeeze() for lab in label]
        else:
            assert is_list_of(label, torch.Tensor), (
                'Only support \'int\', \'np.ndarray\' and \'torch.Tensor\' '
                'type for list input. But receives '
                f'\'{[type(lab) for lab in label]}\'')
            label = [lab.squeeze() for lab in label]
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

    # Check whether is LongTensor (torch.int64) and shape like [bz, ]
    assert label_batch.dtype == torch.int64, (
        'Input label cannot convert to torch.LongTensor (torch.int64). Please '
        'check your input.')
    assert label_batch.ndim == 1, (
        'Input label must shape like \'[num_batches, ]\', but shape like '
        f'({label_batch.shape})')

    # Check the label if `num_classes` is passed. Ignore `num_batches` because
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
