# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine.structures import BaseDataElement, LabelData
from torch import Tensor

from .pixel_data import PixelData


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
    between different components.

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

    @property
    def gt_img(self) -> PixelData:
        """This is the function to fetch gt_img in PixelData.

        Returns:
            PixelData: data element
        """
        return self._gt_img

    @gt_img.setter
    def gt_img(self, value: PixelData):
        """This is the function used to set gt_img in PixelData.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_gt_img', dtype=PixelData)

    @gt_img.deleter
    def gt_img(self):
        """This is the function to fetch gt_img."""
        del self._gt_img

    @property
    def gt_samples(self) -> 'EditDataSample':
        """This is the function to fetch gt_samples.

        Returns:
            EditDataSample: gt samples.
        """
        return self._gt_samples

    @gt_samples.setter
    def gt_samples(self, value: 'EditDataSample'):
        """This is the function to set gt_samples.

        Args:
            value (EditDataSample): gt samples.
        """
        self.set_field(value, '_gt_samples', dtype=EditDataSample)

    @gt_samples.deleter
    def gt_samples(self):
        """This is the function to delete gt_samples."""
        del self._gt_samples

    @property
    def noise(self) -> torch.Tensor:
        """This is the function to fetch noise.

        Returns:
            torch.Tensor: noise.
        """
        return self._noise

    @noise.setter
    def noise(self, value: PixelData):
        """This is the function to set noise.

        Args:
            value (PixelData): noise.
        """
        self.set_field(value, '_noise', dtype=torch.Tensor)

    @noise.deleter
    def noise(self):
        """This is the functionto delete noise."""
        del self._noise

    @property
    def pred_img(self) -> PixelData:
        """This is the function to fetch pred_img in PixelData.

        Returns:
            PixelData: data element
        """
        return self._pred_img

    @pred_img.setter
    def pred_img(self, value: PixelData):
        """This is the function to set the value of pred_img in PixelData.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_pred_img', dtype=PixelData)

    @pred_img.deleter
    def pred_img(self):
        """This is the function to fetch pred_img."""
        del self._pred_img

    @property
    def fake_img(self) -> Union[PixelData, Tensor]:
        """This is the function to fetch fake_img.

        Returns:
            Union[PixelData, Tensor]: The fake img.
        """
        return self._fake_img

    @fake_img.setter
    def fake_img(self, value: Union[PixelData, Tensor]):
        """This is the function to set fake_img.

        Args:
            value (Union[PixelData, Tensor]): The value of fake img.
        """
        assert isinstance(value, (PixelData, Tensor))
        if isinstance(value, PixelData):
            self.set_field(value, '_fake_img', dtype=PixelData)
        else:
            self.set_field(value, '_fake_img', dtype=Tensor)

    @fake_img.deleter
    def fake_img(self):
        """This is the function to delete fake_img."""
        del self._fake_img

    @property
    def img_lq(self) -> PixelData:
        """This is the function to fetch img_lq in PixelData.

        Returns:
            PixelData:  data element
        """
        return self._img_lq

    @img_lq.setter
    def img_lq(self, value: PixelData):
        """This is the function to set img_lq in PixelData.

        Args:
            value (PixelData): data element
        """
        self.set_field(value, '_img_lq', dtype=PixelData)

    @img_lq.deleter
    def img_lq(self):
        """This is the function to delete img_lq."""
        del self._img_lq

    @property
    def ref_img(self) -> PixelData:
        """This is the function to fetch ref_img.

        Returns:
            PixelData:  data element
        """
        return self._ref_img

    @ref_img.setter
    def ref_img(self, value: PixelData):
        """This is the function to set the value of ref_img.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_ref_img', dtype=PixelData)

    @ref_img.deleter
    def ref_img(self):
        """This is the function to fetch ref_img."""
        del self._ref_img

    @property
    def ref_lq(self) -> PixelData:
        """This is the function to fetch ref_lq.

        Returns:
            PixelData:  data element
        """
        return self._ref_lq

    @ref_lq.setter
    def ref_lq(self, value: PixelData):
        """This is the function to set the value of ref_lq.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_ref_lq', dtype=PixelData)

    @ref_lq.deleter
    def ref_lq(self):
        """This is the function to delete ref_lq."""
        del self._ref_lq

    @property
    def gt_unsharp(self) -> PixelData:
        """This is the function to fetch gt_unsharp in PixelData.

        Returns:
            PixelData:  data element
        """
        return self._gt_unsharp

    @gt_unsharp.setter
    def gt_unsharp(self, value: PixelData):
        """This is the function to set the value of gt_unsharp.

        Args:
            value (PixelData): base element
        """
        self.set_field(value, '_gt_unsharp', dtype=PixelData)

    @gt_unsharp.deleter
    def gt_unsharp(self):
        """This is the function to delete gt_unsharp."""
        del self._gt_unsharp

    @property
    def mask(self) -> PixelData:
        """This is the function to fetch mask.

        Returns:
            PixelData:  data element
        """
        return self._mask

    @mask.setter
    def mask(self, value: Union[PixelData, Tensor]):
        """This is the function to set the value of mask.

        Args:
            value (Union[PixelData, Tensor]):  data element
        """
        assert isinstance(value, (PixelData, Tensor))
        if isinstance(value, PixelData):
            self.set_field(value, '_mask', dtype=PixelData)
        else:
            self.set_field(value, '_mask', dtype=Tensor)

    @mask.deleter
    def mask(self):
        """This is the function to delete mask."""
        del self._mask

    @property
    def gt_heatmap(self) -> PixelData:
        """This is the function to fetch gt_heatmap.

        Returns:
            PixelData:  data element
        """
        return self._gt_heatmap

    @gt_heatmap.setter
    def gt_heatmap(self, value: PixelData):
        """This is the function to set the value of gt_heatmap.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_gt_heatmap', dtype=PixelData)

    @gt_heatmap.deleter
    def gt_heatmap(self):
        """This is the function to delete gt_heatmap."""
        del self._gt_heatmap

    @property
    def pred_heatmap(self) -> PixelData:
        """This is the function to fetch pred_heatmap.

        Returns:
            PixelData:  data element
        """
        return self._pred_heatmap

    @pred_heatmap.setter
    def pred_heatmap(self, value: PixelData):
        """This is the function to set the value of pred_heatmap.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_pred_heatmap', dtype=PixelData)

    @pred_heatmap.deleter
    def pred_heatmap(self):
        """This is the function to fetch pred_heatmap."""
        del self._pred_heatmap

    @property
    def trimap(self) -> PixelData:
        """This is the function to fetch trimap.

        Returns:
            PixelData:  data element
        """
        return self._trimap

    @trimap.setter
    def trimap(self, value: PixelData):
        """This is the function to set the value of trimap.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_trimap', dtype=PixelData)

    @trimap.deleter
    def trimap(self):
        """This is the function to delete trimap."""
        del self._trimap

    @property
    def gt_alpha(self) -> PixelData:
        """This is the function to fetch gt_alpha.

        Returns:
            PixelData:  data element
        """
        return self._gt_alpha

    @gt_alpha.setter
    def gt_alpha(self, value: PixelData):
        """This is the function to set the value of gt_alpha.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_gt_alpha', dtype=PixelData)

    @gt_alpha.deleter
    def gt_alpha(self):
        """This is the function to delete gt_alpha."""
        del self._gt_alpha

    @property
    def pred_alpha(self) -> PixelData:
        """This is the function to fetch pred_alpha.

        Returns:
            PixelData:  data element
        """
        return self._pred_alpha

    @pred_alpha.setter
    def pred_alpha(self, value: PixelData):
        """This is the function to set the value of pred_alpha.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_pred_alpha', dtype=PixelData)

    @pred_alpha.deleter
    def pred_alpha(self):
        """This is the function to delete pred_alpha."""
        del self._pred_alpha

    @property
    def gt_fg(self) -> PixelData:
        """This is the function to fetch gt_fg.

        Returns:
            PixelData:  data element
        """
        return self._gt_fg

    @gt_fg.setter
    def gt_fg(self, value: PixelData):
        """This is the function to set the value of gt_fg.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_gt_fg', dtype=PixelData)

    @gt_fg.deleter
    def gt_fg(self):
        """This is the function to delete gt_fg."""
        del self._gt_fg

    @property
    def pred_fg(self) -> PixelData:
        """This is the function to fetch pred_fg.

        Returns:
            PixelData: _description_
        """
        return self._pred_fg

    @pred_fg.setter
    def pred_fg(self, value: PixelData):
        """This is the function to set the value of pred_fg in PixelData.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_pred_fg', dtype=PixelData)

    @pred_fg.deleter
    def pred_fg(self):
        """This is the function to delete pred_fg."""
        del self._pred_fg

    @property
    def gt_bg(self) -> PixelData:
        """This is the function to fetch gt_bg.

        Returns:
            PixelData: data element
        """
        return self._gt_bg

    @gt_bg.setter
    def gt_bg(self, value: PixelData):
        """This is the function to set the value of gt_bg in PixelData.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_gt_bg', dtype=PixelData)

    @gt_bg.deleter
    def gt_bg(self):
        """This is the function to delete gt_bg."""
        del self._gt_bg

    @property
    def pred_bg(self) -> PixelData:
        """This is the function to fetch pred_bg in PixelData.

        Returns:
            PixelData:  data element
        """
        return self._pred_bg

    @pred_bg.setter
    def pred_bg(self, value: PixelData):
        """This is the function to set the value of pred_bg in PixelData.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_pred_bg', dtype=PixelData)

    @pred_bg.deleter
    def pred_bg(self):
        """This is the function to fetch pred_bg."""
        del self._pred_bg

    @property
    def gt_merged(self) -> PixelData:
        """This is the function to fetch gt_merged in PixelData.

        Returns:
            PixelData: _description_
        """
        return self._gt_merged

    @gt_merged.setter
    def gt_merged(self, value: PixelData):
        """This is the function to set gt_merged in PixelDate.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_gt_merged', dtype=PixelData)

    @gt_merged.deleter
    def gt_merged(self):
        """This is the function to fetch gt_merged."""
        del self._gt_merged

    @property
    def sample_model(self) -> str:
        """This is the function to fetch sample model.

        Returns:
            str: Mode of Sample model.
        """
        return self._sample_model

    @sample_model.setter
    def sample_model(self, value: str):
        """This is the function to set sample model.

        Args:
            value (str): The mode of sample model.
        """
        self.set_field(value, '_sample_model', dtype=str)

    @sample_model.deleter
    def sample_model(self):
        """This is the function to delete sample model."""
        del self._sample_model

    @property
    def ema(self) -> 'EditDataSample':
        """This is the function to fetch ema results.

        Returns:
            EditDataSample: Results of the ema model.
        """
        return self._ema

    @ema.setter
    def ema(self, value: 'EditDataSample'):
        """This is the function to set ema results.

        Args:
            value (EditDataSample): Results of the ema model.
        """
        self.set_field(value, '_ema', dtype=EditDataSample)

    @ema.deleter
    def ema(self):
        """This is the function to delete ema results."""
        del self._ema

    @property
    def orig(self) -> 'EditDataSample':
        """This is the function to fetch original results.

        Returns:
            EditDataSample: Results of the ema model.
        """
        return self._orig

    @orig.setter
    def orig(self, value: 'EditDataSample'):
        """This is the function to set ema results.

        Args:
            value (EditDataSample): Results of the ema model.
        """
        self.set_field(value, '_orig', dtype=EditDataSample)

    @orig.deleter
    def orig(self):
        """This is the function to delete ema results."""
        del self._orig

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
        """This is the function to delete gt label."""
        del self._gt_label
