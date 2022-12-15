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
class EditDataPreprocessor(ImgDataPreprocessor):
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
                 non_concentate_keys: Optional[Tuple[str, List[str]]] = None,
                 pad_mode: str = 'constant',
                 only_norm_gt_in_training: bool = False,
                 input_view: Optional[tuple] = None,
                 output_view: Optional[tuple] = None):

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

        self.pad_mode = pad_mode
        self.pad_size_dict = dict()
        self.only_norm_gt_in_training = only_norm_gt_in_training
        self.input_view = input_view
        self.output_view = output_view

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

    def _norm_and_conversion(self, inputs: Tensor) -> Tensor:
        """Norm and conversion the color channel order.

        Args:
            inputs (Tensor): The input tensor

        Returns:
            Tensor: Tensor after normalization and color channel conversion.
        """
        if self._channel_conversion:
            if inputs.ndim == 4:
                inputs = inputs[:, [2, 1, 0], ...]
            else:
                inputs = inputs[[2, 1, 0], ...]
        inputs = inputs.float()
        if self._enable_normalize:
            if self.input_view is None:
                target_shape = [1 for _ in range(inputs.ndim - 3)] + [-1, 1, 1]
            else:
                target_shape = self.input_view
            mean = self.mean.view(target_shape)
            std = self.std.view(target_shape)
            inputs = (inputs - mean) / std
        return inputs

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
        inputs = self._norm_and_conversion(inputs)
        h, w = inputs.shape[2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        batch_inputs = F.pad(inputs, (0, pad_w, 0, pad_h), self.pad_mode,
                             self.pad_value)

        return batch_inputs

    def _preprocess_image_list(self,
                               tensor_list: List[Tensor]) -> Tuple[Tensor]:
        """Preprocess a list of images.

        Returns:
            Tuple[Tensor]: Stacked tensor and padded size.
        """
        dim = tensor_list[0].dim()
        assert all([
            tensor.ndim == dim for tensor in tensor_list
        ]), ('Expected the dimensions of all tensors must be the same, '
             f'but got {[tensor.ndim for tensor in tensor_list]}')

        num_img = len(tensor_list)
        all_sizes: torch.Tensor = torch.Tensor(
            [tensor.shape for tensor in tensor_list])
        max_sizes = torch.ceil(
            torch.max(all_sizes, dim=0)[0] /
            self.pad_size_divisor) * self.pad_size_divisor
        padded_sizes = max_sizes - all_sizes
        # The dim of channel and frame index should not be padded.
        padded_sizes[:, :-2] = 0
        if padded_sizes.sum() == 0:
            stacked_tensor = torch.stack(tensor_list)
            stacked_tensor = self._norm_and_conversion(stacked_tensor)
            return stacked_tensor, padded_sizes

        # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
        # it means that padding the last dim with 1(left) 2(right), padding the
        # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite
        # of the `padded_sizes`. Therefore, the `padded_sizes` needs to be
        # reversed, and only odd index of pad should be assigned to keep
        # padding "right" and "bottom".
        pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
        pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
        batch_tensor = []
        for idx, tensor in enumerate(tensor_list):
            paded_tensor = F.pad(tensor, tuple(pad[idx].tolist()),
                                 self.pad_mode, self.pad_value)
            batch_tensor.append(paded_tensor)
        stacked_tensor = torch.stack(batch_tensor)
        stacked_tensor = self._norm_and_conversion(stacked_tensor)
        return stacked_tensor, padded_sizes

    def _preprocess_dict_inputs(self, batch_inputs: dict) -> dict:
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

                    if k not in self._NON_IMAGE_KEYS:
                        inputs, pad_size = self._preprocess_image_list(inputs)
                        self.pad_size_dict[k] = pad_size

                    batch_inputs[k] = inputs
            elif isinstance(inputs, Tensor) and k not in self._NON_IMAGE_KEYS:
                batch_inputs[k] = self._preprocess_image_tensor(inputs)

        return batch_inputs

    def process_data_sample(self, data_samples: List, training) -> list:
        for data_sample in data_samples:
            if training or not self.only_norm_gt_in_training:
                if not hasattr(data_sample, 'gt_img'):
                    # all data samples should have the same attribute, if
                    # `gt_img` is not found, directly return
                    break
                # NOTE: EditDataPreprocessor only handle gt_img,
                data_sample.gt_img.data = (
                    (data_sample.gt_img.data - self.mean[0]) / self.std[0])
        return data_samples

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
        _batch_data_samples = data.get('data_samples', None)

        # process input
        if isinstance(_batch_inputs, torch.Tensor):
            _batch_inputs = self._preprocess_image_tensor(_batch_inputs)
            _batch_inputs = {'img': _batch_inputs}  # tensor -> dict
            self.pad_size_dict['img'] = None
        elif is_list_of(_batch_inputs, torch.Tensor):
            _batch_inputs, pad_size = self._preprocess_image_list(
                _batch_inputs)
            _batch_inputs = {'img': _batch_inputs}  # tensor -> dict
            self.pad_size_dict['img'] = pad_size
        elif isinstance(_batch_inputs, dict):
            _batch_inputs = self._preprocess_dict_inputs(_batch_inputs)
        elif is_list_of(_batch_inputs, dict):
            # convert list of dict to dict of list
            keys = _batch_inputs[0].keys()
            dict_input = {k: [inp[k] for inp in _batch_inputs] for k in keys}
            _batch_inputs = self._preprocess_dict_inputs(dict_input)
        else:
            raise ValueError('Only support following inputs types: '
                             '\'torch.Tensor\', \'List[torch.Tensor]\', '
                             '\'dict\', \'List[dict]\'. But receive '
                             f'\'{type(_batch_inputs)}\'.')
        data['inputs'] = _batch_inputs

        # process data samples
        if _batch_data_samples:
            _batch_data_samples = self.process_data_sample(
                _batch_data_samples, training)
        data.setdefault('data_samples', _batch_data_samples)

        return data

    def destructor(self, batch_tensor: torch.Tensor, target_key: str = 'img'):
        """Destructor of data processor. Destruct padding, normalization and
        dissolve batch.

        Args:
            batch_tensor (Tensor): Batched output.

        Returns:
            Tensor: Destructed output.
        """
        # TODO: do not know why only destructor as the first sample
        # Do not dissolve batch,
        # all tensor will be de-padded by a same size
        # De pad by the first sample
        assert target_key in self.pad_size_dict, (
            f'Target key \'{target_key}\' cannot be found in saved '
            '\'pad_size_dict\'. Please check whether you have called '
            '\'_preprocess_image_list\' properly.')

        pad_info = self.pad_size_dict[target_key]
        if pad_info is not None:
            padded_h, padded_w = pad_info[0][-2:]
            padded_h = int(padded_h)
            padded_w = int(padded_w)
            h, w = batch_tensor.shape[-2:]
            batch_tensor = batch_tensor[..., :h - padded_h, :w - padded_w]

        # decide norm with self._enable_norm key
        if self._enable_normalize:
            if self.output_view is None:
                target_shape = [1 for _ in range(batch_tensor.ndim - 3)
                                ] + [-1, 1, 1]
            else:
                target_shape = self.output_view
            mean = self.mean.view(target_shape)
            std = self.std.view(target_shape)
            batch_tensor = batch_tensor * std + mean

        # NOTE: should we add clamp here?
        batch_tensor = batch_tensor.clamp_(0, 255)

        return batch_tensor
