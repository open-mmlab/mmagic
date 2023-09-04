# Copyright (c) OpenMMLab. All rights reserved.
import math
from logging import WARNING
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine import print_log
from mmengine.model import ImgDataPreprocessor
from mmengine.utils import is_seq_of
from torch import Tensor

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList

CastData = Union[tuple, dict, DataSample, Tensor, list]


@MODELS.register_module()
class DataPreprocessor(ImgDataPreprocessor):
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
        mean (Sequence[float or int], float or int, optional): The pixel mean
            of image channels. Noted that normalization operation is performed
            *after channel order conversion*. If it is not specified, images
            will not be normalized. Defaults None.
        std (Sequence[float or int], float or int, optional): The pixel
            standard deviation of image channels. Noted that normalization
            operation is performed *after channel order conversion*. If it is
            not specified, images will not be normalized. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        pad_mode (str): Padding mode for ``torch.nn.functional.pad``.
            Defaults to 'constant'.
        non_image_keys (List[str] or str): Keys for fields that not need to be
            processed (padding, channel conversion and normalization) as
            images. If not passed, the keys in :attr:`_NON_IMAGE_KEYS` will be
            used. This argument will only work when `inputs` is dict or list
            of dict. Defaults to None.
        non_concatenate_keys (List[str] or str): Keys for fields that not need
            to be concatenated. If not passed, the keys in
            :attr:`_NON_CONCATENATE_KEYS` will be used. This argument will only
            work when `inputs` is dict or list of dict. Defaults to None.
        output_channel_order (str, optional): The desired image channel order
            of output the data preprocessor. This is also the desired input
            channel order of model (and this most likely to be the output
            order of model). If not passed, no channel order conversion will
            be performed. Defaults to None.
        data_keys (List[str] or str): Keys to preprocess in data samples.
            Defaults to 'gt_img'.
        input_view (tuple, optional): The view of input tensor. This
            argument maybe deleted in the future. Defaults to None.
        output_view (tuple, optional): The view of output tensor. This
            argument maybe deleted in the future. Defaults to None.
        stack_data_sample (bool): Whether stack a list of data samples to one
            data sample. Only support with input data samples are
            `DataSamples`. Defaults to True.
    """
    _NON_IMAGE_KEYS = ['noise']
    _NON_CONCATENATE_KEYS = ['num_batches', 'mode', 'sample_kwargs', 'eq_cfg']

    def __init__(self,
                 mean: Union[Sequence[Union[float, int]], float, int] = 127.5,
                 std: Union[Sequence[Union[float, int]], float, int] = 127.5,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mode: str = 'constant',
                 non_image_keys: Optional[Tuple[str, List[str]]] = None,
                 non_concentate_keys: Optional[Tuple[str, List[str]]] = None,
                 output_channel_order: Optional[str] = None,
                 data_keys: Union[List[str], str] = 'gt_img',
                 input_view: Optional[tuple] = None,
                 output_view: Optional[tuple] = None,
                 stack_data_sample=True):

        if not isinstance(mean, (list, tuple)) and mean is not None:
            mean = [mean]
        if not isinstance(std, (list, tuple)) and std is not None:
            std = [std]

        super().__init__(mean, std, pad_size_divisor, pad_value)
        # get channel order
        assert (output_channel_order is None
                or output_channel_order in ['RGB', 'BGR']), (
                    'Only support \'RGB\', \'BGR\' or None for '
                    '\'output_channel_order\', but receive '
                    f'\'{output_channel_order}\'.')
        self.output_channel_order = output_channel_order

        # add user defined keys
        if non_image_keys is not None:
            if not isinstance(non_image_keys, list):
                non_image_keys = [non_image_keys]
            self._NON_IMAGE_KEYS += non_image_keys
        if non_concentate_keys is not None:
            if not isinstance(non_concentate_keys, list):
                non_concentate_keys = [non_concentate_keys]
            self._NON_CONCATENATE_KEYS += non_concentate_keys

        self.pad_mode = pad_mode
        self.pad_size_dict = dict()
        if data_keys is not None and not isinstance(data_keys, list):
            self.data_keys = [data_keys]
        else:
            self.data_keys = data_keys

        # TODO: can be removed since only be used in LIIF
        self.input_view = input_view
        self.output_view = output_view

        self._done_padding = False  # flag for padding checking
        self._conversion_warning_raised = False  # flag for conversion warning

        self.stack_data_sample = stack_data_sample

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

    @staticmethod
    def _parse_channel_index(inputs) -> int:
        """Parse channel index of inputs."""
        channel_index_mapping = {2: 1, 3: 0, 4: 1, 5: 2}
        if isinstance(inputs, dict):
            ndim = inputs['fake_img'].ndim
            assert ndim in channel_index_mapping, (
                'Only support (H*W, C), (C, H, W), (N, C, H, W) or '
                '(N, t, C, H, W) inputs. But received '
                f'\'({inputs.shape})\'.')
            channel_index = channel_index_mapping[ndim]
        else:
            assert inputs.ndim in channel_index_mapping, (
                'Only support (H*W, C), (C, H, W), (N, C, H, W) or '
                '(N, t, C, H, W) inputs. But received '
                f'\'({inputs.shape})\'.')
            channel_index = channel_index_mapping[inputs.ndim]

        return channel_index

    def _parse_channel_order(self,
                             key: str,
                             inputs: Tensor,
                             data_sample: Optional[DataSample] = None) -> str:
        channel_index = self._parse_channel_index(inputs)
        if isinstance(inputs, dict):
            num_color_channels = inputs['fake_img'].shape[channel_index]
        else:
            num_color_channels = inputs.shape[channel_index]

        # data sample is None, attempt to infer from input tensor
        if data_sample is None:
            if num_color_channels == 1:
                return 'single'
            else:
                # default as BGR
                return 'BGR'

        # data sample is not None, infer from metainfo
        channel_order_key = 'gt_channel_order' if key == 'gt_img' \
            else f'{key}_channel_order'
        color_type_key = 'gt_color_type' if key == 'gt_img' \
            else f'{key}_color_type'

        # TODO: raise warning here, we can build a dict which fields are keys
        # have been parsed.
        color_flag = data_sample.metainfo.get(color_type_key, None)
        channel_order = data_sample.metainfo.get(channel_order_key, None)

        # handle stacked data sample, refers to `DataSample.stack`
        if isinstance(color_flag, list):
            assert all([c == color_flag[0] for c in color_flag])
            color_flag = color_flag[0]
        if isinstance(channel_order, list):
            assert all([c == channel_order[0] for c in channel_order])
            channel_order = channel_order[0]

        # NOTE: to handle inputs such as Y, users may modify the following code
        if color_flag == 'grayscale':
            assert num_color_channels == 1
            return 'single'
        elif color_flag == 'unchanged':
            # if inputs is not None:
            return 'single' if num_color_channels == 1 else 'BGR'
        else:
            # inference from channel_order
            if channel_order:
                return channel_order
            else:
                # no channel order, infer from num channels
                return 'single' if num_color_channels == 1 else 'BGR'

    def _parse_batch_channel_order(self, key: str, inputs: Sequence,
                                   data_samples: Optional[Sequence[DataSample]]
                                   ) -> str:
        """Parse channel order of inputs in batch."""

        assert len(inputs) == len(data_samples)
        batch_inputs_orders = [
            self._parse_channel_order(key, inp, data_sample)
            for inp, data_sample in zip(inputs, data_samples)
        ]
        inputs_order = batch_inputs_orders[0]

        # security checking for channel order
        assert all([
            inputs_order == order for order in batch_inputs_orders
        ]), (f'Channel order ({batch_inputs_orders}) of input targets '
             f'(\'{key}\') are inconsistent.')

        return inputs_order

    def _update_metainfo(self,
                         padding_info: Tensor,
                         channel_order_info: Optional[dict] = None,
                         data_samples: Optional[SampleList] = None
                         ) -> SampleList:
        """Update `padding_info` and `channel_order` to metainfo of.

        *a batch of `data_samples`*. For channel order, we consider same field
        among data samples share the same channel order. Therefore
        `channel_order` is passed as a dict, which key and value are field
        name and corresponding channel order. For padding info, we consider
        padding info is same among all field of a sample, but can vary between
        samples. Therefore, we pass `padding_info` as Tensor shape like
        (B, 1, 1).

        Args:
            padding_info (Tensor): The padding info of each sample. Shape
                like (B, 1, 1).
            channel_order (dict, Optional): The channel order of target field.
                Key and value are field name and corresponding channel order
                respectively.
            data_samples (List[DataSample], optional): The data samples to
                be updated. If not passed, will initialize a list of empty data
                samples. Defaults to None.

        Returns:
            List[DataSample]: The updated data samples.
        """
        n_samples = padding_info.shape[0]
        if data_samples is None:
            data_samples = [DataSample() for _ in range(n_samples)]
        else:
            assert len(data_samples) == n_samples, (
                f'The length of \'data_samples\'({len(data_samples)}) and '
                f'\'padding_info\'({n_samples}) are inconsistent. Please '
                'check your inputs.')

        # update padding info
        for pad_size, data_sample in zip(padding_info, data_samples):
            data_sample.set_metainfo({'padding_size': pad_size})

        # update channel order
        if channel_order_info is not None:
            for data_sample in data_samples:
                for key, channel_order in channel_order_info.items():
                    data_sample.set_metainfo(
                        {f'{key}_output_channel_order': channel_order})

        self._done_padding = padding_info.sum() != 0
        return data_samples

    def _do_conversion(self,
                       inputs: Tensor,
                       inputs_order: str = 'BGR',
                       target_order: Optional[str] = None
                       ) -> Tuple[Tensor, str]:
        """Conduct channel order conversion for *a batch of inputs*, and return
        the converted inputs and order after conversion.

        inputs_order:
            * RGB / RGB: Convert to target order.
            * SINGLE: Do not change
        """
        if (target_order is None
                or inputs_order.upper() == target_order.upper()):
            # order is not changed, return the input one
            return inputs, inputs_order

        def conversion(inputs, channel_index):
            if inputs.shape[channel_index] == 4:
                new_index = [2, 1, 0, 3]
            else:
                new_index = [2, 1, 0]

            # do conversion
            inputs = torch.index_select(
                inputs, channel_index,
                torch.LongTensor(new_index).to(inputs.device))
            return inputs

        channel_index = self._parse_channel_index(inputs)

        if inputs_order.upper() in ['RGB', 'BGR']:
            inputs = conversion(inputs, channel_index)
            return inputs, target_order
        elif inputs_order.upper() == 'SINGLE':
            if not self._conversion_warning_raised:
                print_log(
                    'Cannot convert inputs with \'single\' channel order '
                    f'to \'output_channel_order\' ({self.output_channel_order}'
                    '). Return without conversion.', 'current', WARNING)
                self._conversion_warning_raised = True
            return inputs, inputs_order
        else:
            raise ValueError(f'Unsupported inputs order \'{inputs_order}\'.')

    def _do_norm(self,
                 inputs: Tensor,
                 do_norm: Optional[bool] = None) -> Tensor:

        do_norm = self._enable_normalize if do_norm is None else do_norm

        if do_norm:
            if self.input_view is None:
                if inputs.ndim == 2:  # special case for (H*W, C) tensor
                    target_shape = [1, -1]
                else:
                    target_shape = [1 for _ in range(inputs.ndim - 3)
                                    ] + [-1, 1, 1]
            else:
                target_shape = self.input_view
            mean = self.mean.view(target_shape)
            std = self.std.view(target_shape)

            # shape checking to avoid broadcast a single channel tensor to 3
            channel_idx = self._parse_channel_index(inputs)
            n_channel_inputs = inputs.shape[channel_idx]
            n_channel_mean = mean.shape[channel_idx]
            n_channel_std = std.shape[channel_idx]
            assert n_channel_mean == 1 or n_channel_mean == n_channel_inputs
            assert n_channel_std == 1 or n_channel_std == n_channel_inputs

            inputs = (inputs - mean) / std
        return inputs

    def _preprocess_image_tensor(self,
                                 inputs: Tensor,
                                 data_samples: Optional[SampleList] = None,
                                 key: str = 'img'
                                 ) -> Tuple[Tensor, SampleList]:
        """Preprocess a batch of image tensor and update metainfo to
        corresponding data samples.

        Args:
            inputs (Tensor): Image tensor with shape (C, H, W), (N, C, H, W) or
                (N, t, C, H, W) to preprocess.
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.
            key (str): The key of image tensor in data samples.
                Defaults to 'img'.

        Returns:
            Tuple[Tensor, List[DataSample]]: The preprocessed image tensor
                and updated data samples.
        """
        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(inputs.shape[0])]

        assert inputs.dim() in [
            3, 4, 5
        ], ('The input of `_preprocess_image_tensor` should be a (C, H, W), '
            '(N, C, H, W) or (N, t, C, H, W)tensor, but got a tensor with '
            f'shape: {inputs.shape}')
        channel_order = self._parse_batch_channel_order(
            key, inputs, data_samples)
        inputs, output_channel_order = self._do_conversion(
            inputs, channel_order, self.output_channel_order)
        inputs = self._do_norm(inputs)
        h, w = inputs.shape[-2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        batch_inputs = F.pad(inputs, (0, pad_w, 0, pad_h), self.pad_mode,
                             self.pad_value)

        padding_size = torch.FloatTensor((0, pad_h, pad_w))[None, ...]
        padding_size = padding_size.repeat(inputs.shape[0], 1)
        data_samples = self._update_metainfo(padding_size,
                                             {key: output_channel_order},
                                             data_samples)
        return batch_inputs, data_samples

    def _preprocess_image_list(self,
                               tensor_list: List[Tensor],
                               data_samples: Optional[SampleList],
                               key: str = 'img') -> Tuple[Tensor, SampleList]:
        """Preprocess a list of image tensor and update metainfo to
        corresponding data samples.

        Args:
            tensor_list (List[Tensor]): Image tensor list to be preprocess.
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.
            key (str): The key of tensor list in data samples.
                Defaults to 'img'.

        Returns:
            Tuple[Tensor, List[DataSample]]: The preprocessed image tensor
                and updated data samples.
        """
        if not data_samples:  # none or empty list
            data_samples = [DataSample() for _ in range(len(tensor_list))]

        channel_order = self._parse_batch_channel_order(
            key, tensor_list, data_samples)
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
            stacked_tensor, output_channel_order = self._do_conversion(
                stacked_tensor, channel_order, self.output_channel_order)
            stacked_tensor = self._do_norm(stacked_tensor)
            data_samples = self._update_metainfo(padding_sizes,
                                                 {key: output_channel_order},
                                                 data_samples)
            return stacked_tensor, data_samples

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
        stacked_tensor, output_channel_order = self._do_conversion(
            stacked_tensor, channel_order, self.output_channel_order)
        stacked_tensor = self._do_norm(stacked_tensor)
        data_samples = self._update_metainfo(padding_sizes,
                                             {key: output_channel_order},
                                             data_samples)
        # return stacked_tensor, padding_sizes
        return stacked_tensor, data_samples

    def _preprocess_dict_inputs(self,
                                batch_inputs: dict,
                                data_samples: Optional[SampleList] = None
                                ) -> Tuple[dict, SampleList]:
        """Preprocess dict type inputs.

        Args:
            batch_inputs (dict): Input dict.
            data_samples (List[DataSample], optional): The data samples
                of corresponding inputs. If not passed, a list of empty data
                samples will be initialized to save metainfo. Defaults to None.

        Returns:
            Tuple[dict, List[DataSample]]: The preprocessed dict and
                updated data samples.
        """
        pad_size_dict = dict()
        for k, inputs in batch_inputs.items():
            # handle concentrate for values in list
            if isinstance(inputs, list):
                if k in self._NON_CONCATENATE_KEYS:
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
                        # preprocess as image
                        inputs, data_samples = self._preprocess_image_list(
                            inputs, data_samples, k)
                        pad_size_dict[k] = [
                            data.metainfo.get('padding_size')
                            for data in data_samples
                        ]
                    else:
                        # only stack
                        inputs = torch.stack(inputs)

                    batch_inputs[k] = inputs

            elif isinstance(inputs, Tensor) and k not in self._NON_IMAGE_KEYS:
                batch_inputs[k], data_samples = \
                    self._preprocess_image_tensor(inputs, data_samples, k)
                pad_size_dict[k] = [
                    data.metainfo.get('padding_size') for data in data_samples
                ]

        # NOTE: we only support all key shares the same padding size
        if pad_size_dict:
            padding_sizes = list(pad_size_dict.values())[0]
            padding_key = list(pad_size_dict.keys())[0]
            for idx, tar_size in enumerate(padding_sizes):
                for k, sizes in pad_size_dict.items():
                    if (tar_size != sizes[idx]).any():
                        raise ValueError(
                            f'All fields of a data sample should share the '
                            'same padding size, but got different size for '
                            f'\'{k}\'(\'{sizes[idx]}\') and \'{padding_key}\''
                            f'(\'{tar_size}\') at index {idx}.Please check '
                            'your data carefully.')

        return batch_inputs, data_samples

    def _preprocess_data_sample(self, data_samples: SampleList,
                                training: bool) -> DataSample:
        """Preprocess data samples. When `training` is True, fields belong to
        :attr:`self.data_keys` will be converted to
        :attr:`self.output_channel_order` and then normalized by `self.mean`
        and `self.std`. When `training` is False, fields belongs to
        :attr:`self.data_keys` will be attempted to convert to 'BGR' without
        normalization. The corresponding metainfo related to normalization,
        channel order conversion will be updated to data sample as well.

        Args:
            data_samples (List[DataSample]): A list of data samples to
                preprocess.
            training (bool): Whether in training mode.

        Returns:
            list: The list of processed data samples.
        """
        if not training:
            # set default order to BGR in test stage
            target_order, do_norm = 'BGR', False
        else:
            # norm in training, conversion as default (None)
            target_order, do_norm = self.output_channel_order, True

        for data_sample in data_samples:
            if not self.data_keys:
                break
            for key in self.data_keys:
                if not hasattr(data_sample, key):
                    # do not raise error here
                    print_log(f'Cannot find key \'{key}\' in data sample.',
                              'current', WARNING)
                    break

                data = data_sample.get(key)
                data_channel_order = self._parse_channel_order(
                    key, data, data_sample)
                data, channel_order = self._do_conversion(
                    data, data_channel_order, target_order)
                data = self._do_norm(data, do_norm)
                data_sample.set_data({f'{key}': data})
                data_process_meta = {
                    f'{key}_enable_norm': self._enable_normalize,
                    f'{key}_output_channel_order': channel_order,
                    f'{key}_mean': self.mean,
                    f'{key}_std': self.std
                }
                data_sample.set_metainfo(data_process_meta)

        if self.stack_data_sample:
            assert is_seq_of(data_samples, DataSample), (
                'Only support \'stack_data_sample\' for DataSample '
                'object. Please refer to \'DataSample.stack\'.')
            return DataSample.stack(data_samples)
        return data_samples

    def forward(self, data: dict, training: bool = False) -> dict:
        """Performs normalization、padding and channel order conversion.

        Args:
            data (dict): Input data to process.
            training (bool): Whether to in training mode. Default: False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        _batch_data_samples = data.get('data_samples', None)

        # process input
        if isinstance(_batch_inputs, torch.Tensor):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_image_tensor(
                    _batch_inputs, _batch_data_samples)
        elif is_seq_of(_batch_inputs, torch.Tensor):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_image_list(
                    _batch_inputs, _batch_data_samples)
        elif isinstance(_batch_inputs, dict):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_dict_inputs(
                    _batch_inputs, _batch_data_samples)
        elif is_seq_of(_batch_inputs, dict):
            # convert list of dict to dict of list
            keys = _batch_inputs[0].keys()
            dict_input = {k: [inp[k] for inp in _batch_inputs] for k in keys}
            _batch_inputs, _batch_data_samples = \
                self._preprocess_dict_inputs(
                    dict_input, _batch_data_samples)
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

        data['data_samples'] = _batch_data_samples

        return data

    def destruct(self,
                 outputs: Tensor,
                 data_samples: Union[SampleList, DataSample, None] = None,
                 key: str = 'img') -> Union[list, Tensor]:
        """Destruct padding, normalization and convert channel order to BGR if
        could. If `data_samples` is a list, outputs will be destructed as a
        batch of tensor. If `data_samples` is a `DataSample`, `outputs` will be
        destructed as a single tensor.

        Before feed model outputs to visualizer and evaluator, users should
        call this function for model outputs and inputs.

        Use cases:

        >>> # destruct model outputs.
        >>> # model outputs share the same preprocess information with inputs
        >>> # ('img') therefore use 'img' as key
        >>> feats = self.forward_tensor(inputs, data_samples, **kwargs)
        >>> feats = self.data_preprocessor.destruct(feats, data_samples, 'img')

        >>> # destruct model inputs for visualization
        >>> for idx, data_sample in enumerate(data_samples):
        >>>     destructed_input = self.data_preprocessor.destruct(
        >>>         inputs[idx], data_sample, key='img')
        >>>     data_sample.set_data({'input': destructed_input})

        Args:
            outputs (Tensor): Tensor to destruct.
            data_samples (Union[SampleList, DataSample], optional): Data
                samples (or data sample) corresponding to `outputs`.
                Defaults to None
            key (str): The key of field in data sample. Defaults to 'img'.

        Returns:
            Union[list, Tensor]: Destructed outputs.
        """
        # NOTE: only support passing tensor sample, if the output of model is
        # a dict, users should call this manually.
        # Since we do not know whether the outputs is image tensor.
        _batch_outputs = self._destruct_norm_and_conversion(
            outputs, data_samples, key)
        _batch_outputs = self._destruct_padding(_batch_outputs, data_samples)
        _batch_outputs = _batch_outputs.clamp_(0, 255)
        return _batch_outputs

    def _destruct_norm_and_conversion(self, batch_tensor: Tensor,
                                      data_samples: Union[SampleList,
                                                          DataSample, None],
                                      key: str) -> Tensor:
        """De-norm and de-convert channel order. Noted that, we de-norm first,
        and then de-conversion, since mean and std used in normalization is
        based on channel order after conversion.

        Args:
            batch_tensor (Tensor): Tensor to destruct.
            data_samples (Union[SampleList, DataSample], optional): Data
                samples (or data sample) corresponding to `outputs`.
            key (str): The key of field in data sample.

        Returns:
            Tensor: Destructed tensor.
        """

        output_key = f'{key}_output'
        # get channel order from data sample
        if isinstance(data_samples, list):
            inputs_order = self._parse_batch_channel_order(
                output_key, batch_tensor, data_samples)
        else:
            inputs_order = self._parse_channel_order(output_key, batch_tensor,
                                                     data_samples)
        if self._enable_normalize:
            if self.output_view is None:
                if batch_tensor.ndim == 2:  # special case for (H*W, C) tensor
                    target_shape = [1, -1]
                else:
                    target_shape = [1 for _ in range(batch_tensor.ndim - 3)
                                    ] + [-1, 1, 1]
            else:
                target_shape = self.output_view
            mean = self.mean.view(target_shape)
            std = self.std.view(target_shape)
            batch_tensor = batch_tensor * std + mean

        # convert output to 'BGR' if able
        batch_tensor, _ = self._do_conversion(
            batch_tensor, inputs_order=inputs_order, target_order='BGR')

        return batch_tensor

    def _destruct_padding(self,
                          batch_tensor: Tensor,
                          data_samples: Union[SampleList, DataSample, None],
                          same_padding: bool = True) -> Union[list, Tensor]:
        """Destruct padding of the input tensor.

        Args:
            batch_tensor (Tensor): Tensor to destruct.
            data_samples (Union[SampleList, DataSample], optional): Data
                samples (or data sample) corresponding to `outputs`. If
            same_padding (bool): Whether all samples will un-padded with the
                padding info of the first sample, and return a stacked
                un-padded tensor. Otherwise each sample will be unpadded with
                padding info saved in corresponding data samples, and return a
                list of un-padded tensor, since each un-padded tensor may have
                the different shape. Defaults to True.

        Returns:
            Union[list, Tensor]: Destructed outputs.
        """
        # NOTE: If same padding, batch_tensor will un-padded with the padding
        # info # of the first sample and return a Unpadded tensor. Otherwise,
        # input tensor # will un-padded with the corresponding padding info
        # saved in data samples and return a list of tensor.
        if data_samples is None:
            return batch_tensor

        if isinstance(data_samples, list):
            is_batch_data = True
            if 'padding_size' in data_samples[0].metainfo_keys():
                pad_infos = [
                    sample.metainfo['padding_size'] for sample in data_samples
                ]
            else:
                pad_infos = None
        else:
            if 'padding_size' in data_samples.metainfo_keys():
                pad_infos = data_samples.metainfo['padding_size']
            else:
                pad_infos = None
            # NOTE: here we assume padding size in metainfo are saved as tensor
            if not isinstance(pad_infos, list):
                pad_infos = [pad_infos]
                is_batch_data = False
            else:
                is_batch_data = True
            if all([pad_info is None for pad_info in pad_infos]):
                pad_infos = None

        if not is_batch_data:
            batch_tensor = batch_tensor[None, ...]

        if pad_infos is None:
            if self._done_padding:
                print_log(
                    'Cannot find padding information (\'padding_size\') in '
                    'meta info of \'data_samples\'. Please check whether '
                    'you have called \'self.forward\' properly.', 'current',
                    WARNING)
            return batch_tensor if is_batch_data else batch_tensor[0]

        if same_padding:
            # un-pad with the padding info of the first sample
            padded_h, padded_w = pad_infos[0][-2:]
            padded_h, padded_w = int(padded_h), int(padded_w)
            h, w = batch_tensor.shape[-2:]
            batch_tensor = batch_tensor[..., :h - padded_h, :w - padded_w]
            return batch_tensor if is_batch_data else batch_tensor[0]
        else:
            # un-pad with the corresponding padding info
            unpadded_tensors = []
            for idx, pad_info in enumerate(pad_infos):
                padded_h, padded_w = pad_info[-2:]
                padded_h = int(padded_h)
                padded_w = int(padded_w)
                h, w = batch_tensor[idx].shape[-2:]
                unpadded_tensor = batch_tensor[idx][..., :h - padded_h, :w -
                                                    padded_w]
                unpadded_tensors.append(unpadded_tensor)
            return unpadded_tensors if is_batch_data else unpadded_tensors[0]
