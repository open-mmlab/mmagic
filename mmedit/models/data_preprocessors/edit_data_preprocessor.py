# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmedit.registry import MODELS


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
        batch_tensor (Tensor): The 4D-tensor or 5D-tensor. \
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


@MODELS.register_module()
class EditDataPreprocessor(BaseDataPreprocessor):
    """Basic data pre-processor used for collating and copying data to the
    target device in mmediting.

    ``EditDataPreprocessor`` performs data pre-processing according to the
    following steps:

    - Collates the data sampled from dataloader.
    - Copies data to the target device.
    - Stacks the input tensor at the first dimension.

    and post-processing of the output tensor of model.

    TODO: Most editing methods have crop inputs to a same size, batched padding
        will be faster.

    Args:
        mean (Sequence[float or int]): The pixel mean of R, G, B channels.
            Defaults to (0, 0, 0). If ``mean`` and ``std`` are not
            specified, ImgDataPreprocessor will normalize images to [0, 1].
        std (Sequence[float or int]): The pixel standard deviation of R, G, B
            channels. (255, 255, 255). If ``mean`` and ``std`` are not
            specified, ImgDataPreprocessor will normalize images to [0, 1].
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        input_view (Tuple | List): Tensor view of mean and std for input
            (without batch). Defaults to (-1, 1, 1) for (C, H, W).
        output_view (Tuple | List | None): Tensor view of mean and std for
            output (without batch). If None, output_view=input_view.
            Defaults: None.
        pad_args (dict): Args of F.pad. Default: dict().
    """

    def __init__(
        self,
        mean: Sequence[Union[float, int]] = (0, 0, 0),
        std: Sequence[Union[float, int]] = (255, 255, 255),
        pad_size_divisor: int = 1,
        input_view=(-1, 1, 1),
        output_view=None,
        pad_args: dict = dict(),
        # non_image_keys: Optional[Tuple[str, List[str]]] = None,
        # non_concentate_keys: Optional[Tuple[str, List[str]]] = None,
    ) -> None:

        super().__init__()

        assert len(mean) == 3 or len(mean) == 1, (
            'The length of mean should be 1 or 3 to be compatible with RGB '
            f'or gray image, but got {len(mean)}')
        assert len(std) == 3 or len(std) == 1, (
            'The length of mean should be 1 or 3 to be compatible with RGB '
            f'or gray image, but got {len(std)}')

        # reshape mean and std for input (without batch).
        self.register_buffer('input_mean',
                             torch.tensor(mean).view(input_view), False)
        self.register_buffer('input_std',
                             torch.tensor(std).view(input_view), False)

        # reshape mean and std for batched output.
        if output_view is None:
            output_view = input_view
        batched_output_view = [1] + list(output_view)  # add batch dim
        self.register_buffer('outputs_mean',
                             torch.tensor(mean).view(batched_output_view),
                             False)
        self.register_buffer('outputs_std',
                             torch.tensor(std).view(batched_output_view),
                             False)

        self.pad_size_divisor = pad_size_divisor
        self.pad_args = pad_args
        self.padded_sizes = None
        self.norm_input_flag = None  # If input is normalized to [0, 1]

    def forward(
        self,
        data: Sequence[dict],
        training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Pre-process the data into the model input format.

        After the data pre-processing of :meth:`collate_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Default: False.

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """

        # inputs, batch_data_samples = self.collate_data(data)
        data = super().forward(data=data, training=training)
        inputs, batch_data_samples = data['inputs'], data['data_samples']

        # Check if input is normalized to [0, 1]
        self.norm_input_flag = (inputs[0].max() <= 1)

        # Normalization.
        inputs = [(_input - self.input_mean) / self.input_std
                  for _input in inputs]

        # Pad and stack Tensor.
        inputs, self.padded_sizes = stack_batch(inputs, self.pad_size_divisor,
                                                self.pad_args)

        if training:
            for data_sample in batch_data_samples:
                data_sample.gt_img.data = (
                    (data_sample.gt_img.data - self.outputs_mean[0]) /
                    self.outputs_std[0])

        data['inputs'] = inputs
        data['data_samples'] = batch_data_samples
        return data

    def destructor(self, batch_tensor: torch.Tensor):
        """Destructor of data processor. Destruct padding, normalization and
        dissolve batch.

        Args:
            batch_tensor (Tensor): Batched output.

        Returns:
            Tensor: Destructed output.
        """

        # De-normalization
        batch_tensor = batch_tensor * self.outputs_std + self.outputs_mean

        # Do not dissolve batch,
        # all tensor will be de-padded by a same size
        # De pad by the first sample
        padded_h, padded_w = self.padded_sizes[0][-2:]
        padded_h = int(padded_h)
        padded_w = int(padded_w)
        h, w = batch_tensor.shape[-2:]
        batch_tensor = batch_tensor[..., :h - padded_h, :w - padded_w]

        assert self.norm_input_flag is not None, (
            'Please kindly run `forward` before running `destructor`')
        if self.norm_input_flag:
            batch_tensor *= 255
        batch_tensor = batch_tensor.clamp_(0, 255)

        return batch_tensor
