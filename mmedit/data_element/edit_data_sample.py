# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.data import BaseDataElement

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
        - ``ignored_data``: Data to be ignored during
            training/testing.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.data import PixelData
         >>> from mmedit.core import EditDataSample
         >>> data_sample = EditDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3))
         >>> gt_img = PixelData(metainfo=img_meta)
         >>> gt_img = torch.rand((3, 800, 1196))
         >>> data_sample.gt_img = gt_img
         >>> assert 'img_shape' in data_sample.gt_img.metainfo_keys()
         >>> len(data_sample.gt_img)
         5
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_img: <PixelData(

                    META INFORMATION
                    pad_shape: (800, 1216, 3)
                    img_shape: (800, 1196, 3)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> pred_img = PixelData(metainfo=img_meta)
         >>> pred_img.bboxes = torch.rand((5, 4))
         >>> pred_img.scores = torch.rand((5,))
         >>> data_sample.pred_img = pred_img
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_img: <PixelData(

                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    pad_shape: (800, 1216, 3)

                    DATA FIELDS
                    labels: tensor([0.2627, 0.3778, 0.2038, 0.3375, 0.9851])
                    bboxes: tensor([[0.2193, 0.1166, 0.5360, 0.7132],
                                [0.9293, 0.8853, 0.5077, 0.6340],
                                [0.0620, 0.4339, 0.4820, 0.0469],
                                [0.9636, 0.5789, 0.4534, 0.1950],
                                [0.0142, 0.4320, 0.5696, 0.9699]])
                ) at 0x7f0b7b8d6190>
            pred_img: <PixelData(

                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    pad_shape: (800, 1216, 3)

                    DATA FIELDS
                    bboxes: tensor([[0.0246, 0.9718, 0.7703, 0.7454],
                                [0.7734, 0.9903, 0.4460, 0.9821],
                                [0.0435, 0.7106, 0.0547, 0.8913],
                                [0.7212, 0.6108, 0.8398, 0.4018],
                                [0.2552, 0.1137, 0.1374, 0.4659]])
                    scores: tensor([0.5802, 0.9886, 0.9108, 0.2662, 0.9374])
                ) at 0x7f0b7b8d61c0>
        ) at 0x7f0b7b8d6880>
    """

    @property
    def gt_img(self) -> PixelData:
        return self._gt_img

    @gt_img.setter
    def gt_img(self, value: PixelData):
        self.set_field(value, '_gt_img', dtype=PixelData)

    @gt_img.deleter
    def gt_img(self):
        del self._gt_img

    @property
    def pred_img(self) -> PixelData:
        return self._pred_img

    @pred_img.setter
    def pred_img(self, value: PixelData):
        self.set_field(value, '_pred_img', dtype=PixelData)

    @pred_img.deleter
    def pred_img(self):
        del self._pred_img

    @property
    def ref_img(self) -> PixelData:
        return self._ref_img

    @ref_img.setter
    def ref_img(self, value: PixelData):
        self.set_field(value, '_ref_img', dtype=PixelData)

    @ref_img.deleter
    def ref_img(self):
        del self._ref_img

    @property
    def mask(self) -> PixelData:
        return self._mask

    @mask.setter
    def mask(self, value: PixelData):
        self.set_field(value, '_mask', dtype=PixelData)

    @mask.deleter
    def mask(self):
        del self._mask

    @property
    def gt_heatmap(self) -> PixelData:
        return self._gt_heatmap

    @gt_heatmap.setter
    def gt_heatmap(self, value: PixelData):
        self.set_field(value, '_gt_heatmap', dtype=PixelData)

    @gt_heatmap.deleter
    def gt_heatmap(self):
        del self._gt_heatmap

    @property
    def pred_heatmap(self) -> PixelData:
        return self._pred_heatmap

    @pred_heatmap.setter
    def pred_heatmap(self, value: PixelData):
        self.set_field(value, '_pred_heatmap', dtype=PixelData)

    @pred_heatmap.deleter
    def pred_heatmap(self):
        del self._pred_heatmap

    @property
    def trimap(self) -> PixelData:
        return self._trimap

    @trimap.setter
    def trimap(self, value: PixelData):
        self.set_field(value, '_trimap', dtype=PixelData)

    @trimap.deleter
    def trimap(self):
        del self._trimap

    @property
    def gt_alpha(self) -> PixelData:
        return self._gt_alpha

    @gt_alpha.setter
    def gt_alpha(self, value: PixelData):
        self.set_field(value, '_gt_alpha', dtype=PixelData)

    @gt_alpha.deleter
    def gt_alpha(self):
        del self._gt_alpha

    @property
    def pred_alpha(self) -> PixelData:
        return self._pred_alpha

    @pred_alpha.setter
    def pred_alpha(self, value: PixelData):
        self.set_field(value, '_pred_alpha', dtype=PixelData)

    @pred_alpha.deleter
    def pred_alpha(self):
        del self._pred_alpha

    @property
    def gt_fg(self) -> PixelData:
        return self._gt_fg

    @gt_fg.setter
    def gt_fg(self, value: PixelData):
        self.set_field(value, '_gt_fg', dtype=PixelData)

    @gt_fg.deleter
    def gt_fg(self):
        del self._gt_fg

    @property
    def pred_fg(self) -> PixelData:
        return self._pred_fg

    @pred_fg.setter
    def pred_fg(self, value: PixelData):
        self.set_field(value, '_pred_fg', dtype=PixelData)

    @pred_fg.deleter
    def pred_fg(self):
        del self._pred_fg

    @property
    def gt_bg(self) -> PixelData:
        return self._gt_bg

    @gt_bg.setter
    def gt_bg(self, value: PixelData):
        self.set_field(value, '_gt_bg', dtype=PixelData)

    @gt_bg.deleter
    def gt_bg(self):
        del self._gt_bg

    @property
    def pred_bg(self) -> PixelData:
        return self._pred_bg

    @pred_bg.setter
    def pred_bg(self, value: PixelData):
        self.set_field(value, '_pred_bg', dtype=PixelData)

    @pred_bg.deleter
    def pred_bg(self):
        del self._pred_bg

    @property
    def ignored_data(self) -> BaseDataElement:
        return self._ignored_data

    @ignored_data.setter
    def ignored_data(self, value: BaseDataElement):
        self.set_field(value, '_ignored_data', dtype=BaseDataElement)

    @ignored_data.deleter
    def ignored_data(self):
        del self._ignored_data
