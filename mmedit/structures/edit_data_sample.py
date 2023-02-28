# Copyright (c) OpenMMLab. All rights reserved.
from collections import abc
from copy import deepcopy
from itertools import chain
from numbers import Number
from typing import Any, Sequence, Union
from warnings import warn

import mmengine
import numpy as np
import torch
from mmengine.structures import BaseDataElement, LabelData

from mmedit.utils import all_to_tensor


def format_label(value: Union[torch.Tensor, np.ndarray, Sequence, int],
                 num_classes: int = None) -> LabelData:
    """Convert label of various python types to :obj:`mmengine.LabelData`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.
        num_classes (int, optional): The number of classes. If not None, set
            it to the metainfo. Defaults to None.

    Returns:
        :obj:`mmengine.LabelData`: The foramtted label data.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')

    metainfo = {}
    if num_classes is not None:
        metainfo['num_classes'] = num_classes
        if value.max() >= num_classes:
            raise ValueError(f'The label data ({value}) should not '
                             f'exceed num_classes ({num_classes}).')
    label = LabelData(label=value, metainfo=metainfo)
    return label


def is_stacked_var(var: Any) -> bool:
    """Check whether input variable is a stacked variable and can be split.

    Args:
        var (Any): The input variable to check.

    Returns:
        bool: Whether input variable is a stacked variable.
    """
    if isinstance(var, EditDataSample) and var.is_stacked:
        return True
    if isinstance(var, torch.Tensor):
        return True
    if isinstance(var, abc.Sequence) and not isinstance(var, str):
        return True
    return False


class EditDataSample(BaseDataElement):
    """A data structure interface of MMEditing. They are used as interfaces
    between different components, e.g., model, visualizer, evaluator, etc.
    Typically, EditDataSample contains all the information and data from
    ground-truth and predictions.

    `EditDataSample` inherits from `BaseDataElement`. See more details in:
      https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html
      Specifically, an instance of BaseDataElement consists of two components,
      - ``metainfo``, which contains some meta information,
        e.g., `img_shape`, `img_id`, etc.
      - ``data``, which contains the data used in the loop.

    The attributes in ``EditDataSample`` are divided into several parts:

    - ``gt_img``: Ground truth image(s).
    - ``pred_img``: Image(s) of model predictions.
    - ``ref_img``: Reference image(s).
    - ``mask``: Mask in Inpainting.
    - ``trimap``: Trimap in Matting.
    - ``gt_alpha``: Ground truth alpha image in Matting.
    - ``pred_alpha``: Predicted alpha image in Matting.
    - ``gt_fg``: Ground truth foreground image in Matting.
    - ``pred_fg``: Predicted foreground image in Matting.
    - ``gt_bg``: Ground truth background image in Matting.
    - ``pred_bg``: Predicted background image in Matting.
    - ``gt_merged``: Ground truth merged image in Matting.

    Examples::

         >>> import torch
         >>> import numpy as np
         >>> from mmedit.structures import EditDataSample
         >>> img_meta = dict(img_shape=(800, 1196, 3))
         >>> img = torch.rand((3, 800, 1196))
         >>> data_sample = EditDataSample(gt_img=img, metainfo=img_meta)
         >>> assert 'img_shape' in data_sample.metainfo_keys()
        <EditDataSample(

            META INFORMATION
            img_shape: (800, 1196, 3)

            DATA FIELDS
            gt_img: tensor(...)
        ) at 0x1f6a5a99a00>
    """

    # source_key_in_results: target_key_in_metainfo
    META_KEYS = {
        'img_path': 'img_path',
        'gt_path': 'gt_path',
        'merged_path': 'merged_path',
        'trimap_path': 'trimap_path',
        'ori_shape': 'ori_shape',
        'img_shape': 'img_shape',
        'ori_merged_shape': 'ori_merged_shape',
        'ori_trimap_shape': 'ori_trimap_shape',
        'trimap_channel_order': 'trimap_channel_order',
        'empty_box': 'empty_box',
        'ori_img_shape': 'ori_img_shape',
        'ori_gt_shape': 'ori_gt_shape',
        'img_channel_order': 'img_channel_order',
        'gt_channel_order': 'gt_channel_order',
        'gt_color_type': 'gt_color_type',
        'img_color_type': 'img_color_type',
        'sample_idx': 'sample_idx',
        'num_input_frames': 'num_input_frames',
        'num_output_frames': 'num_output_frames',
        # for LIIF
        'coord': 'coord',
        'cell': 'cell',
    }

    # source_key_in_results: target_key_in_datafield
    DATA_KEYS = {
        'gt': 'gt_img',
        'gt_label': 'gt_label',
        'gt_heatmap': 'gt_heatmap',
        'gt_unsharp': 'gt_unsharp',
        'merged': 'gt_merged',
        'ori_alpha': 'ori_alpha',
        'fg': 'gt_fg',
        'bg': 'gt_bg',
        'gt_rgb': 'gt_rgb',
        'alpha': 'gt_alpha',
        'img_lq': 'img_lq',
        'ref': 'ref_img',
        'ref_lq': 'ref_lq',
        'mask': 'mask',
        'trimap': 'trimap',
        'gray': 'gray',
        'cropped_img': 'cropped_img',
        'pred_img': 'pred_img',
        'ori_trimap': 'ori_trimap'
    }

    _is_stacked = False

    @property
    def is_stacked(self):
        return self._is_stacked

    def set_predefined_data(self, data: dict) -> None:
        """set or change pre-defined key-value pairs in ``data_field`` by
        parameter ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """

        metainfo = {
            self.META_KEYS[k]: v
            for (k, v) in data.items() if k in self.META_KEYS
        }
        self.set_metainfo(metainfo)

        data = {
            self.DATA_KEYS[k]: v
            for (k, v) in data.items() if k in self.DATA_KEYS
        }
        self.set_tensor_data(data)

    def set_tensor_data(self, data: dict) -> None:
        """convert input data to tensor, and then set or change key-value pairs
        in ``data_field`` by parameter ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data,
                          dict), f'data should be a `dict` but got {data}'
        for k, v in data.items():
            if k == 'gt_label':
                self.set_gt_label(v)
            else:
                self.set_field(all_to_tensor(v), k, dtype=torch.Tensor)

    def set_gt_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'EditDataSample':
        """Set label of ``gt_label``."""
        label = format_label(value, self.get('num_classes'))
        if 'gt_label' in self:
            self.gt_label.label = label.label
        else:
            self.gt_label = label
        return self

    @property
    def gt_label(self):
        """This the function to fetch gt label.

        Returns:
            LabelData: gt label.
        """
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData):
        """This is the function to set gt label.

        Args:
            value (LabelData): gt label.
        """
        self.set_field(value, '_gt_label', dtype=LabelData)

    @gt_label.deleter
    def gt_label(self):
        """Delete gt label."""
        del self._gt_label

    @classmethod
    def stack(cls,
              data_samples: Sequence['EditDataSample']) -> 'EditDataSample':
        """Stack a list of data samples to one. All tensor fields will be
        stacked at first dimension. Otherwise the values will be saved in a
        list.

        Args:
            data_samples (Sequence['EditDataSample']): A sequence of
                `EditDataSample` to stack.

        Returns:
            EditDataSample: The stacked data sample.
        """
        # 0. check if is empty

        # 1. check key consistency
        keys = data_samples[0].keys()
        assert all([data.keys() == keys for data in data_samples])

        meta_keys = data_samples[0].metainfo_keys()
        assert all(
            [data.metainfo_keys() == meta_keys for data in data_samples])

        # 2. stack data
        stacked_data_sample = EditDataSample()
        for k in keys:
            values = [getattr(data, k) for data in data_samples]
            # 3. check type consistent
            value_type = type(values[0])
            assert all([type(val) == value_type for val in values])

            # 4. stack
            if isinstance(values[0], torch.Tensor):
                stacked_value = torch.stack(values)
            elif isinstance(values[0], LabelData):
                labels = [data.label for data in values]
                values = torch.stack(labels)
                stacked_value = LabelData(label=values)
            else:
                stacked_value = values
            stacked_data_sample.set_field(stacked_value, k)

        # 5. stack metainfo
        for k in meta_keys:
            values = [data.metainfo[k] for data in data_samples]
            stacked_data_sample.set_metainfo({k: values})

        stacked_data_sample._is_stacked = True
        return stacked_data_sample

    def split(self,
              allow_nonseq_value: bool = False) -> Sequence['EditDataSample']:
        """Split a sequence of data sample in the first dimension.

        Args:
            allow_nonseq_value (bool): Whether allow non-sequential data in
                split operation. If True, non-sequential data will be copied
                for all split data samples. Otherwise, an error will be
                raised. Defaults to False.

        Returns:
            Sequence[EditDataSample]: The list of data samples after splitting.
        """
        assert self.is_stacked, (
            'Only support to call \'split\' for stacked data sample. Please '
            'refer to \'EditDataSample.stack\' for more details.')
        # 1. split
        data_sample_list = [EditDataSample() for _ in range(len(self))]
        for k in self.all_keys():
            if k == '_is_stacked':
                continue
            stacked_value = self.get(k)
            if isinstance(stacked_value, torch.Tensor):
                # split tensor shape like (N, *shape) to N (*shape) tensors
                values = [v for v in stacked_value]
            elif isinstance(stacked_value, LabelData):
                # split tensor shape like (N, *shape) to N (*shape) tensors
                labels = [l_ for l_ in stacked_value.label]
                values = [LabelData(label=l_) for l_ in labels]
            elif isinstance(stacked_value, EditDataSample):
                assert stacked_value.is_stacked, (
                    f'Field \'{k}\' is \'EditDataSample\' type but '
                    '\'is_stacked\' is False. Do not support split unstacked '
                    'EditDataSample.')
                values = stacked_value.split()
            else:
                if is_stacked_var(stacked_value):
                    values = stacked_value
                elif allow_nonseq_value:
                    values = [deepcopy(stacked_value)] * len(self)
                else:
                    raise TypeError(
                        f'\'{k}\' is non-sequential data and '
                        '\'allow_nonseq_value\' is False. Please check your '
                        'data sample or set \'allow_nonseq_value\' as True '
                        f'to copy fiedl \'{k}\' for all split data sample.')

            field = 'metainfo' if k in self.metainfo_keys() else 'data'
            for data, v in zip(data_sample_list, values):
                data.set_field(v, k, field_type=field)

        return data_sample_list

    def __len__(self):
        """Get the length of the data sample."""
        if self._is_stacked:
            value_length = []
            for k, v in chain(self.items(), self.metainfo_items()):
                if k == '_is_stacked':
                    continue
                if isinstance(v, LabelData):
                    value_length.append(v.label.shape[0])
                elif is_stacked_var(v):
                    value_length.append(len(v))
                else:
                    continue
            if not value_length:
                warn('Cannot found any element in stacked data sample, '
                     'please check your data sample carefully.')
                return 1
            assert len(list(set(value_length))) == 1
            length = value_length[0]
            return length
        return 1
