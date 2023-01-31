# Copyright (c) OpenMMLab. All rights reserved.
import math
from logging import WARNING
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine import print_log
from mmengine.model import ImgDataPreprocessor
from mmengine.structures import BaseDataElement
from mmengine.utils import is_seq_of
from torch import Tensor

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample

CastData = Union[tuple, dict, BaseDataElement, Tensor, list]


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
        data_keys: keys to preprocess in data samples.
        norm_data_samples_in_testing (bool): Whether normalize data samples in
            testing stage. Defaults to False.
    """
    _NON_IMAGE_KEYS = ['noise']
    _NON_CONCENTATE_KEYS = ['num_batches', 'mode', 'sample_kwargs', 'eq_cfg']

    def __init__(
            self,
            mean: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
            std: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
            pad_size_divisor: int = 1,
            pad_value: Union[float, int] = 0,
            bgr_to_rgb: bool = False,
            rgb_to_bgr: bool = False,
            non_image_keys: Optional[Tuple[str, List[str]]] = None,
            non_concentate_keys: Optional[Tuple[str, List[str]]] = None,
            pad_mode: str = 'constant',
            # norm_data_samples_in_testing: bool = False,
            #  input_keys: Union[List[str], str] = 'inputs',
            output_channel_order: str = 'unchanged',
            data_keys: Union[List[str], str] = 'gt_img',
            input_view: Optional[tuple] = None,
            output_view: Optional[tuple] = None):

        super().__init__(mean, std, pad_size_divisor, pad_value, bgr_to_rgb,
                         rgb_to_bgr)
        # get color order
        # if bgr_to_rgb:
        #     input_color_order, output_color_order = 'bgr', 'rgb'
        # elif rgb_to_bgr:
        #     input_color_order, output_color_order = 'rgb', 'bgr'
        # else:
        #     # 'bgr' order as default
        #     input_color_order = output_color_order = 'bgr'
        # self.input_color_order = input_color_order
        # self.output_color_order = output_color_order
        self.output_color_order = output_channel_order

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
        # self.norm_data_samples_in_testing = norm_data_samples_in_testing
        # self.input_keys = input_keys if isinstance(input_keys,
        #                                            list) else [input_keys]
        self.data_keys = data_keys if isinstance(data_keys,
                                                 list) else [data_keys]
        self.input_view = input_view
        self.output_view = output_view

        self._done_padding = False  # flag for padding checking
        self._conversion_warning_raised = False  # flag for conversion warning

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

    def _do_conversion(self,
                       inputs: Tensor,
                       inputs_order: str = 'BGR',
                       target_order: Optional[str] = None
                       ) -> Tuple[Tensor, str]:
        """return converted inputs and order after conversion."""
        # inputs: [CHW] or [NCHW]
        target_order = self.output_color_order \
            if target_order is None else target_order

        if target_order == 'unchanged' or inputs_order == target_order:
            return inputs, target_order

        channel_index_mapping = {3: 0, 4: 1, 5: 2}
        assert inputs.ndim in channel_index_mapping, (
            'Only support (C, H, W), (N, C, H, W) or (N, t, C, H, W) '
            f'inputs. But received \'({inputs.shape})\'.')
        channel_index = channel_index_mapping[inputs.ndim]

        if inputs.shape[channel_index] == 1:
            if not self._conversion_warning_raised:
                print_log(
                    'Cannot convert single channel input to '
                    f'\'output_color_order\'({self.output_color_order})',
                    'current', WARNING)
                self._conversion_warning_raised = True
            return inputs, 'unchanged'

        if inputs.shape[channel_index] == 4:
            new_index = [2, 1, 0, 3]
        else:
            new_index = [2, 1, 0]

        # do conversion
        inputs = torch.index_select(
            inputs, channel_index,
            torch.IntTensor(new_index).to(inputs.device))

        return inputs, target_order

    def _do_norm(self,
                 inputs: Tensor,
                 do_norm: Optional[bool] = None) -> Tensor:

        do_norm = self._enable_normalize if do_norm is None else do_norm

        if do_norm:
            if self.input_view is None:
                target_shape = [1 for _ in range(inputs.ndim - 3)] + [-1, 1, 1]
            else:
                target_shape = self.input_view
            mean = self.mean.view(target_shape)
            std = self.std.view(target_shape)
            inputs = (inputs - mean) / std
        return inputs

    def _norm_and_conversion(self,
                             inputs: Tensor,
                             do_norm: Optional[bool] = None,
                             do_conversion: Optional[bool] = None) -> Tensor:
        """Norm and conversion the color channel order.

        Args:
            inputs (Tensor): The input tensor.
            do_norm (bool): Whether normalize the input tensor. If False,
                :attr:`self._enable_normalize` will be ignored and
                normalization will be skipped.
            do_conversion (bool): Whether apply color channel conversion the
                input tensor. If False, :attr:`self._channel_conversion` will
                be ignored and channel conversion will be skipped.

        Returns:
            Tensor: Tensor after normalization and color channel conversion.
        """
        do_norm = self._enable_normalize if do_norm is None else do_norm
        do_conversion = self._channel_conversion \
            if do_conversion is None else do_conversion

        if do_conversion:
            new_index = [2, 1, 0]
            if inputs.ndim == 4:
                # handle RGBA
                if inputs.shape[1] == 4:
                    new_index += [3]
                inputs = inputs[:, [*new_index], ...]
            else:
                # handle RGBA
                if inputs.shape[0] == 4:
                    new_index += [3]
                inputs = inputs[[*new_index], ...]
        inputs = inputs.float()
        if do_norm:
            if self.input_view is None:
                target_shape = [1 for _ in range(inputs.ndim - 3)] + [-1, 1, 1]
            else:
                target_shape = self.input_view
            mean = self.mean.view(target_shape)
            std = self.std.view(target_shape)
            inputs = (inputs - mean) / std
        return inputs

    def _preprocess_image_tensor(self, inputs: Tensor,
                                 channel_order: str) -> Tensor:
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
        # inputs = self._norm_and_conversion(inputs)
        inputs, _ = self._do_conversion(inputs, channel_order)
        inputs = self._do_norm(inputs)
        h, w = inputs.shape[2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        batch_inputs = F.pad(inputs, (0, pad_w, 0, pad_h), self.pad_mode,
                             self.pad_value)

        return batch_inputs

    def _preprocess_image_list(self, tensor_list: List[Tensor],
                               channel_order: str) -> Tuple[Tensor, Tensor]:
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
        padding_sizes = max_sizes - all_sizes
        # The dim of channel and frame index should not be padded.
        padding_sizes[:, :-2] = 0
        if padding_sizes.sum() == 0:
            stacked_tensor = torch.stack(tensor_list)
            # stacked_tensor = self._norm_and_conversion(stacked_tensor)
            stacked_tensor, _ = self._do_conversion(stacked_tensor,
                                                    channel_order)
            stacked_tensor = self._do_norm(stacked_tensor)
            return stacked_tensor, padding_sizes

        # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
        # it means that padding the last dim with 1(left) 2(right), padding the
        # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite
        # of the `padded_sizes`. Therefore, the `padded_sizes` needs to be
        # reversed, and only odd index of pad should be assigned to keep
        # padding "right" and "bottom".
        pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
        pad[:, 1::2] = padding_sizes[:, range(dim - 1, -1, -1)]
        batch_tensor = []
        for idx, tensor in enumerate(tensor_list):
            paded_tensor = F.pad(tensor, tuple(pad[idx].tolist()),
                                 self.pad_mode, self.pad_value)
            batch_tensor.append(paded_tensor)
        stacked_tensor = torch.stack(batch_tensor)
        # stacked_tensor = self._norm_and_conversion(stacked_tensor)
        stacked_tensor, _ = self._do_conversion(stacked_tensor, channel_order)
        stacked_tensor = self._do_norm(stacked_tensor)
        return stacked_tensor, padding_sizes

    def _preprocess_dict_inputs(self, batch_inputs: dict,
                                channel_order) -> dict:
        """Preprocess dict type inputs.

        Args:
            batch_inputs (dict): Input dict.

        Returns:
            dict: Preprocessed dict.
        """
        pad_size_dict = dict()
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
                        inputs, padding_sizes = self._preprocess_image_list(
                            inputs)
                        # self.pad_size_dict[k] = pad_size
                        pad_size_dict[k] = [size for size in padding_sizes]

                    batch_inputs[k] = inputs
            elif isinstance(inputs, Tensor) and k not in self._NON_IMAGE_KEYS:
                batch_inputs[k], padding_sizes = self._preprocess_image_tensor(
                    inputs, channel_order)
                pad_size_dict[k] = [size for size in padding_sizes]

        # NOTE: we only support all key shares the same padding size
        if pad_size_dict:
            padding_size = list(pad_size_dict.values())[0]
            padding_key = list(pad_size_dict.keys())[0]
            for k, v in pad_size_dict.items():
                assert v == padding_size, (
                    'All keys should share the same padding size, but got '
                    f'different size for \'{k}\' (\'{v}\') and '
                    f'\'{padding_key}\' (\'{padding_size}\'). Please check '
                    'your data carefully.')

            return batch_inputs, padding_sizes

        return batch_inputs, None

    def _preprocess_data_sample(self, data_samples: List,
                                training: bool) -> list:
        """Process data samples. Output will be normed and conversed as inputs.

        # NOTE: this function do not handle padding
        # >>> the following docstring is wrong
        If `norm_data_samples_in_training` is True,
        data_samples will only be normed in both training and testing phase.
        Otherwise,
        # <<< the above docstring is wrong

        Args:
            data_samples (List): A list of data samples to process.
            training (bool): Whether in training mode.

        Returns:
            list: The list of processed data samples.
        """
        # import ipdb
        # ipdb.set_trace()
        if not training:
            # set default order to BGR in test stage
            target_order, do_norm = 'BGR', False
        else:
            # norm in training, conversion as default (None)
            target_order, do_norm = self.output_color_order, True
            # do_norm, do_conversion = True, None

        for data_sample in data_samples:
            for key in self.data_keys:
                if not hasattr(data_sample, key):
                    # do not raise error here
                    print_log(f'Cannot find key \'{key}\' in data sample.',
                              'current', WARNING)
                    break

                data = data_sample.get(key)
                channel_order_key = 'gt_channel_order' if key == 'gt_img' \
                    else f'{key}_channel_order'
                # data_channel_order = data_sample.get_metainfo(
                #     f'{key}_color_order', None)
                data_channel_order = data_sample.metainfo.get(
                    channel_order_key, None)
                data, color_order = self._do_conversion(
                    data, data_channel_order, target_order)
                data = self._do_norm(data, do_norm)
                data_sample.set_data({f'{key}': data})
                # data_sample.set_data({
                #     f'{key}':
                #     self._norm_and_conversion(
                #         data_sample.get(key), do_norm, do_conversion)
                # })
                data_process_meta = {
                    f'{key}_enable_norm': self._enable_normalize,
                    # f'{key}_enable_conversion': self._channel_conversion,
                    # save real order
                    f'{key}_output_color_order': color_order,
                    f'{key}_mean': self.mean,
                    f'{key}_std': self.std
                }
                data_sample.set_metainfo(data_process_meta)

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

        # ops, for translation model?
        if _batch_data_samples is not None:
            inputs_channel_order = _batch_data_samples[0].get(
                'img_channel_order', 'bgr')
        else:
            inputs_channel_order = None

        # process input
        if isinstance(_batch_inputs, torch.Tensor):
            _batch_inputs = self._preprocess_image_tensor(
                _batch_inputs, inputs_channel_order)
            _batch_inputs = {'img': _batch_inputs}  # tensor -> dict
            _pad_info = None
        elif is_seq_of(_batch_inputs, torch.Tensor):
            _batch_inputs, _pad_info = self._preprocess_image_list(
                _batch_inputs, inputs_channel_order)
            # _batch_inputs = _batch_inputs
        elif isinstance(_batch_inputs, dict):
            _batch_inputs, _pad_info = self._preprocess_dict_inputs(
                _batch_inputs, inputs_channel_order)
        elif is_seq_of(_batch_inputs, dict):
            # convert list of dict to dict of list
            keys = _batch_inputs[0].keys()
            dict_input = {k: [inp[k] for inp in _batch_inputs] for k in keys}
            _batch_inputs, _pad_info = self._preprocess_dict_inputs(
                dict_input, inputs_channel_order)
        else:
            raise ValueError('Only support following inputs types: '
                             '\'torch.Tensor\', \'List[torch.Tensor]\', '
                             '\'dict\', \'List[dict]\'. But receive '
                             f'\'{type(_batch_inputs)}\'.')
        data['inputs'] = _batch_inputs

        # process data samples
        if _batch_data_samples:
            _batch_data_samples = self._preprocess_data_sample(
                _batch_data_samples, training)

        # update pad_size_dict to data_samples
        if _pad_info is not None:
            # NOTE: ops, we have to init a empty data sample to save
            # padding_size
            if not _batch_data_samples:
                _batch_data_samples = [
                    EditDataSample() for _ in range(len(_pad_info))
                ]
            for pad_size, data_sample in zip(_pad_info, _batch_data_samples):
                data_sample.set_metainfo({'padding_size': pad_size})
            self._done_padding = True  # NOTE: any better way?

        data.setdefault('data_samples', _batch_data_samples)

        return data

    def destructor(
        self,
        outputs: Tensor,
        data_samples: Optional[List[EditDataSample]] = None
    ) -> Union[list, Tensor]:
        # NOTE: only support passing tensor sample, if the output of model is
        # a dict, users should call this manually.
        # Since we do not know whether the outputs is image tensor.
        _batch_outputs = self._destruct_tensor_norm_and_conversion(outputs)
        _batch_outputs = self._destruct_tensor_padding(_batch_outputs,
                                                       data_samples)
        _batch_outputs = _batch_outputs.clamp_(0, 255)
        return _batch_outputs

    def _destruct_tensor_norm_and_conversion(self,
                                             batch_tensor: Tensor) -> Tensor:

        # single channel -> return
        # 3 channel -> check conversion
        # if self.output_color_order == 'RGB':
        #     if batch_tensor.ndim == 4:
        #         batch_tensor = batch_tensor[..., [2, 1, 0]]
        #     else:
        #         batch_tensor = batch_tensor[[2, 1, 0], ...]

        # convert output to 'BGR' if able
        inputs_order = 'BGR' if self.output_color_order is None \
            else self.output_color_order
        batch_tensor, _ = self._do_conversion(
            batch_tensor, inputs_order=inputs_order, target_order='BGR')

        if self._enable_normalize:
            if self.output_view is None:
                target_shape = [1 for _ in range(batch_tensor.ndim - 3)
                                ] + [-1, 1, 1]
            else:
                target_shape = self.output_view
            mean = self.mean.view(target_shape)
            std = self.std.view(target_shape)
            batch_tensor = batch_tensor * std + mean

        return batch_tensor

    def _destruct_tensor_padding(
            self,
            batch_tensor: Tensor,
            data_samples: Optional[List[EditDataSample]] = None,
            same_padding: bool = True) -> Union[list, Tensor]:
        # NOTE: If same padding, batch_tensor will un-padded with the padding
        # info # of the first sample and return a Unpadded tensor. Otherwise,
        # input tensor # will un-padded with the corresponding padding info
        # saved in data samples and return a list of tensor.
        if data_samples is None:
            return batch_tensor

        if not hasattr(data_samples[0], 'padding_size'):
            if self._done_padding:
                print_log(
                    'Cannot find padding information (\'padding_size\') in '
                    'meta info of \'data_samples\'. Please check whether '
                    'you have called \'self.forward\' properly.', 'current',
                    WARNING)
            return batch_tensor

        pad_infos = [
            sample.metainfo['padding_size'] for sample in data_samples
        ]
        if same_padding:
            padded_h, padded_w = pad_infos[0][-2:]
            padded_h, padded_w = int(padded_h), int(padded_w)
            h, w = batch_tensor.shape[-2:]
            batch_tensor = batch_tensor[..., :h - padded_h, :w - padded_w]
            return batch_tensor
        else:
            unpadded_tensors = []
            for idx, pad_info in enumerate(pad_infos):
                padded_h, padded_w = pad_info[-2:]
                padded_h = int(padded_h)
                padded_w = int(padded_w)
                h, w = batch_tensor[idx].shape[-2:]
                unpadded_tensor = batch_tensor[idx][..., :h - padded_h, :w -
                                                    padded_w]
                unpadded_tensors.append(unpadded_tensor)
            return unpadded_tensors
