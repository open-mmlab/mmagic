# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Sequence, Union

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
         >>> from mmedit.structures import EditDataSample, PixelData
         >>> data_sample = EditDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3))
         >>> img = torch.rand((3, 800, 1196))
         >>> gt_img = PixelData(data=img, metainfo=img_meta)
         >>> data_sample.gt_img = gt_img
         >>> assert 'img_shape' in data_sample.gt_img.metainfo_keys()
        <EditDataSample(

            META INFORMATION

            DATA FIELDS
            _gt_img: <PixelData(

                    META INFORMATION
                    img_shape: (800, 1196, 3)

                    DATA FIELDS
                    data: tensor([[[0.8069, 0.4279,  ..., 0.6603, 0.0292],

                                ...,

                                [0.8139, 0.0908,  ..., 0.4964, 0.9672]]])
                ) at 0x1f6ae000af0>
            gt_img: <PixelData(

                    META INFORMATION
                    img_shape: (800, 1196, 3)

                    DATA FIELDS
                    data: tensor([[[0.8069, 0.4279,  ..., 0.6603, 0.0292],

                                ...,

                                [0.8139, 0.0908,  ..., 0.4964, 0.9672]]])
                ) at 0x1f6ae000af0>
        ) at 0x1f6a5a99a00>
    """

    # source_key_in_results: target_key_in_metainfo
    META_KEYS = {
        'img_path': 'img_path',
        'merged_path': 'merged_path',
        'trimap_path': 'trimap_path',
        'ori_shape': 'ori_shape',
        'img_shape': 'img_shape',
        'ori_merged_shape': 'ori_merged_shape',
        'ori_trimap_shape': 'ori_trimap_shape',
        'trimap_channel_order': 'trimap_channel_order',
        'empty_box': 'empty_box'
    }

    # source_key_in_results: target_key_in_datafield
    DATA_KEYS = {
        'gt': 'gt_img',
        'gt_label': 'gt_label',
        'gt_heatmap': 'gt_heatmap',
        'gt_unsharp': 'gt_unsharp',
        'merged': 'gt_merged',
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
                setattr(self, k, all_to_tensor(v))

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
