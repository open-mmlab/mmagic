# Copyright (c) OpenMMLab. All rights reserved.
import copy
from numbers import Number
from typing import List, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.structures import BaseDataElement, LabelData


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
    META_KEYS = ['img_path', 'ori_shape', 'img_shape']

    # source_key_in_results: target_key_in_datafield
    DATA_KEYS = [
        'gt_img', 'gt_label', 'gt_heatmap', 'gt_unsharp', 'gt_merged', 'gt_fg',
        'gt_bg', 'gt_rgb', 'gt_alpha', 'img_lq', 'ref_img', 'ref_lq', 'mask',
        'trimap', 'gray', 'cropped_img', 'pred_img'
    ]

    def _extend_keys(
        self,
        pre_defined_keys: Tuple[List[str], dict, None],
        input_keys: Tuple[List[str], str, None],
    ) -> Tuple[List[str], dict]:
        """Extend pre_defined keys with user-provided keys.

        Args:
            pre_defined_keys (Union[List[str], dict])
        """
        result_keys = copy.deepcopy(pre_defined_keys)
        if input_keys is not None:
            if isinstance(result_keys, list):
                return result_keys.extend(input_keys)
            if isinstance(input_keys, list):
                for k in input_keys:
                    result_keys.update({k: k})
            else:
                result_keys.update({input_keys: input_keys})
        return result_keys

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
