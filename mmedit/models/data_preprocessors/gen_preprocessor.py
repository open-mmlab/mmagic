# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import ImgDataPreprocessor
from mmengine.structures import BaseDataElement
from mmengine.utils import is_list_of
from torch import Tensor

from mmedit.registry import MODELS

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list]


@MODELS.register_module()
class GenDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for generative models. This class provide
    normalization and bgr to rgb conversion for image tensor inputs. The input
    of this classes should be dict which keys are `inputs` and `data_samples`.

    Besides to process tensor `inputs`, this class support dict as `inputs`.
    - If the value is `Tensor` and the corresponding key is not contained in
    :attr:`_NON_IMAGE_KEYS`, it will be processed as image tensor.
    - If the value is `Tensor` and the corresponding key belongs to
    :attr:`_NON_IMAGE_KEYS`, it will not remains unchanged.
    - If value is string or integer, it will not remains unchanged.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of image
            channels. If ``bgr_to_rgb=True`` it means the mean value of R,
            G, B channels. If it is not specified, images will not be
            normalized. Defaults None.
        std (Sequence[float or int], optional): The pixel standard deviation of
            image channels. If ``bgr_to_rgb=True`` it means the standard
            deviation of R, G, B channels. If it is not specified, images will
            not be normalized. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
    """
    _NON_IMAGE_KEYS = ['noise']
    _NON_CONCENTATE_KEYS = ['num_batches', 'mode', 'sample_kwargs', 'eq_cfg']

    def __init__(self,
                 mean: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 std: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_image_keys: Optional[Tuple[str, List[str]]] = None,
                 non_concentate_keys: Optional[Tuple[str, List[str]]] = None):

        super().__init__(mean, std, pad_size_divisor, pad_value, bgr_to_rgb,
                         rgb_to_bgr)
        # get color order
        if bgr_to_rgb:
            input_color_order, output_color_order = 'bgr', 'rgb'
        elif rgb_to_bgr:
            input_color_order, output_color_order = 'rgb', 'bgr'
        else:
            # 'bgr' order as default
            input_color_order = output_color_order = 'bgr'
        self.input_color_order = input_color_order
        self.output_color_order = output_color_order

        # add user defined keys
        if non_image_keys is not None:
            if not isinstance(non_image_keys, list):
                non_image_keys = [non_image_keys]
            self._NON_IMAGE_KEYS += non_image_keys
        if non_concentate_keys is not None:
            if not isinstance(non_concentate_keys, list):
                non_concentate_keys = [non_concentate_keys]
            self._NON_CONCENTATE_KEYS += non_concentate_keys

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, (str, int, float)):
            return data
        return super().cast_data(data)

    def _preprocess_image_tensor(self, inputs: Tensor) -> Tensor:
        """Process image tensor.

        Args:
            inputs (Tensor): List of image tensor to process.

        Returns:
            Tensor: Processed and stacked image tensor.
        """
        assert inputs.dim() == 4, (
            'The input of `_preprocess_image_tensor` should be a NCHW '
            'tensor or a list of tensor, but got a tensor with shape: '
            f'{inputs.shape}')
        if self._channel_conversion:
            inputs = inputs[:, [2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        inputs = inputs.float()
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std
        h, w = inputs.shape[2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        batch_inputs = F.pad(inputs, (0, pad_w, 0, pad_h), 'constant',
                             self.pad_value)

        return batch_inputs

    def process_dict_inputs(self, batch_inputs: dict) -> dict:
        """Preprocess dict type inputs.

        Args:
            batch_inputs (dict): Input dict.

        Returns:
            dict: Preprocessed dict.
        """
        for k, inputs in batch_inputs.items():
            # handle concentrate for values in list
            if isinstance(inputs, list):
                if k in self._NON_CONCENTATE_KEYS:
                    # use the first value
                    assert all([
                        inputs[0] == inp for inp in inputs
                    ]), (f'NON_CONCENTATE_KEY \'{k}\' should be consistency '
                         'among the data list.')
                    batch_inputs[k] = inputs[0]
                else:
                    assert all([
                        isinstance(inp, torch.Tensor) for inp in inputs
                    ]), ('Only support stack list of Tensor in inputs dict. '
                         f'But \'{k}\' is list of \'{type(inputs[0])}\'.')
                    inputs = torch.stack(inputs, dim=0)

                    if k not in self._NON_IMAGE_KEYS:
                        # process as image
                        inputs = self._preprocess_image_tensor(inputs)

                    batch_inputs[k] = inputs
            elif isinstance(inputs, Tensor) and k not in self._NON_IMAGE_KEYS:
                batch_inputs[k] = self._preprocess_image_tensor(inputs)

        return batch_inputs

    def forward(self, data: dict, training: bool = False) -> dict:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Input data to process.
            training (bool): Whether to enable training time augmentation.
                This is ignored for :class:`GenDataPreprocessor`. Defaults to
                False.
        Returns:
            dict: Data in the same format as the model input.
        """

        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        if (isinstance(_batch_inputs, torch.Tensor)
                or is_list_of(_batch_inputs, torch.Tensor)):
            data = super().forward(data, training)
            # pack inputs to a dict
            data['inputs'] = {'img': data['inputs']}
            return data
        elif isinstance(_batch_inputs, dict):
            _batch_inputs = self.process_dict_inputs(_batch_inputs)
        else:
            raise ValueError('')

        data['inputs'] = _batch_inputs
        data.setdefault('data_samples', None)
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
        batch_tensor = batch_tensor * self.std + self.mean
        batch_tensor = batch_tensor.clamp_(0, 255)

        return batch_tensor
