# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement

from .pixel_data import PixelData


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

    Examples:
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
    def mask(self, value: PixelData):
        """This is the function to set the value of mask.

        Args:
            value (PixelData):  data element
        """
        self.set_field(value, '_mask', dtype=PixelData)

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
