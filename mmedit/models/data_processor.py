# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor

from mmedit.registry import MODELS
from .utils import stack_batch


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
            Defaults to (255, 255, 255). If ``mean`` and ``std`` are not
            specified, ImgDataPreprocessor will normalize images to [0, 1].
        std (Sequence[float or int]): The pixel standard deviation of R, G, B
            channels. (0, 0, 0). If ``mean`` and ``std`` are not
            specified, ImgDataPreprocessor will normalize images to [0, 1].
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        input_view (Tuple | List): Tensor view of mean and std for input
            (without batch). Defaults to (-1, 1, 1) for (C, H, W).
        output_view (Tuple | List | None): Tensor view of mean and std for
            output (without batch). If None, output_view=input_view.
            Defaults: None.
        pad_args: Args of F.pad.
    """

    def __init__(
            self,
            mean: Sequence[Union[float, int]] = (0, 0, 0),
            std: Sequence[Union[float, int]] = (255, 255, 255),
            pad_size_divisor: int = 1,
            input_view=(-1, 1, 1),
            output_view=None,
            pad_args: dict = dict(),
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

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """

        inputs, batch_data_samples = self.collate_data(data)

        # Normalization.
        inputs = [(_input - self.input_mean) / self.input_std
                  for _input in inputs]

        # Pad and stack Tensor.
        batch_inputs, self.padded_sizes = stack_batch(inputs,
                                                      self.pad_size_divisor,
                                                      self.pad_args)

        if training:
            for data_sample in batch_data_samples:
                data_sample.gt_img.data = (
                    (data_sample.gt_img.data - self.outputs_mean[0]) /
                    self.outputs_std[0])

        return batch_inputs, batch_data_samples

    def destructor(self, batch_tensor: torch.Tensor):
        """Destructor of data processor.
        Destruct padding, normalization and dissolve batch.

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

        return batch_tensor
