# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmedit.utils.typing import ForwardInputs


def gather_log_vars(log_vars_list: List[Dict[str,
                                             Tensor]]) -> Dict[str, Tensor]:
    """Gather a list of log_vars.
    Args:
        log_vars_list: List[Dict[str, Tensor]]

    Returns:
        Dict[str, Tensor]
    """
    if len(log_vars_list) == 1:
        return log_vars_list[0]

    log_keys = log_vars_list[0].keys()

    log_vars = dict()
    for k in log_keys:
        assert all([k in log_vars for log_vars in log_vars_list
                    ]), (f'\'{k}\' not in some of the \'log_vars\'.')
        log_vars[k] = torch.mean(
            torch.stack([log_vars[k] for log_vars in log_vars_list], dim=0))

    return log_vars


class GANImageBuffer:
    """This class implements an image buffer that stores previously generated
    images.

    This buffer allows us to update the discriminator using a history of
    generated images rather than the ones produced by the latest generator
    to reduce model oscillation.

    Args:
        buffer_size (int): The size of image buffer. If buffer_size = 0,
            no buffer will be created.
        buffer_ratio (float): The chance / possibility  to use the images
            previously stored in the buffer.
    """

    def __init__(self, buffer_size, buffer_ratio=0.5):
        self.buffer_size = buffer_size
        # create an empty buffer
        if self.buffer_size > 0:
            self.img_num = 0
            self.image_buffer = []
        self.buffer_ratio = buffer_ratio

    def query(self, images):
        """Query current image batch using a history of generated images.

        Args:
            images (Tensor): Current image batch without history information.
        """
        if self.buffer_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if the buffer is not full, keep inserting current images
            if self.img_num < self.buffer_size:
                self.img_num = self.img_num + 1
                self.image_buffer.append(image)
                return_images.append(image)
            else:
                use_buffer = np.random.random() < self.buffer_ratio
                # by self.buffer_ratio, the buffer will return a previously
                # stored image, and insert the current image into the buffer
                if use_buffer:
                    random_id = np.random.randint(0, self.buffer_size)
                    image_tmp = self.image_buffer[random_id].clone()
                    self.image_buffer[random_id] = image
                    return_images.append(image_tmp)
                # by (1 - self.buffer_ratio), the buffer will return the
                # current image
                else:
                    return_images.append(image)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images


def get_valid_noise_size(noise_size: Optional[int],
                         generator: Union[Dict, nn.Module]) -> Optional[int]:
    """Get the value of `noise_size` from input, `generator` and check the
    consistency of these values. If no conflict is found, return that value.

    Args:
        noise_size (Optional[int]): `noise_size` passed to
            `BaseGAN_refactor`'s initialize function.
        generator (ModelType): The config or the model of generator.

    Returns:
        int | None: The noise size feed to generator.
    """
    if isinstance(generator, dict):
        model_noise_size = generator.get('noise_size', None)
    else:
        model_noise_size = getattr(generator, 'noise_size', None)

    # get noise_size
    if noise_size is not None and model_noise_size is not None:
        assert noise_size == model_noise_size, (
            'Input \'noise_size\' is unconsistency with '
            f'\'generator.noise_size\'. Receive \'{noise_size}\' and '
            f'\'{model_noise_size}\'.')
    else:
        noise_size = noise_size or model_noise_size

    return noise_size


def get_valid_num_batches(batch_inputs: ForwardInputs) -> int:
    """Try get the valid batch size from inputs.

    - If some values in `batch_inputs` are `Tensor` and 'num_batches' is in
      `batch_inputs`, we check whether the value of 'num_batches' and the the
      length of first dimension of all tensors are same. If the values are not
      same, `AssertionError` will be raised. If all values are the same,
      return the value.
    - If no values in `batch_inputs` is `Tensor`, 'num_batches' must be
      contained in `batch_inputs`. And this value will be returned.
    - If some values are `Tensor` and 'num_batches' is not contained in
      `batch_inputs`, we check whether all tensor have the same length on the
      first dimension. If the length are not same, `AssertionError` will be
      raised. If all length are the same, return the length as batch size.
    - If batch_inputs is a `Tensor`, directly return the length of the first
      dimension as batch size.

    Args:
        batch_inputs (ForwardInputs): Inputs passed to :meth:`forward`.

    Returns:
        int: The batch size of samples to generate.
    """
    if isinstance(batch_inputs, Tensor):
        return batch_inputs.shape[0]

    # get num_batces from batch_inputs
    num_batches_dict = {
        k: v.shape[0]
        for k, v in batch_inputs.items() if isinstance(v, Tensor)
    }
    if 'num_batches' in batch_inputs:
        num_batches_dict['num_batches'] = batch_inputs['num_batches']

    # ensure num_batches is not None
    assert len(num_batches_dict.keys()) > 0, (
        'Cannot get \'num_batches\' form preprocessed input '
        f'(\'{batch_inputs}\').')

    # ensure all num_batches are same
    num_batches = list(num_batches_dict.values())[0]
    assert all([
        bz == num_batches for bz in num_batches_dict.values()
    ]), ('\'num_batches\' is inconsistency among the preprocessed input. '
         f'\'num_batches\' parsed resutls: {num_batches_dict}')

    return num_batches


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
