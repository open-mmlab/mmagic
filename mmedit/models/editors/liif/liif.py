# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch

from mmedit.models.__base__ import BaseEditModel
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData


@MODELS.register_module()
class LIIF(BaseEditModel):
    """LIIF model for single image super-resolution.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        pretrained (str): Path for pretrained model. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """

    def forward_tensor(self, batch_inputs, data_samples=None, **kwargs):
        """Forward tensor.
            Returns result of simple forward.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        coord = torch.stack([
            data_sample.metainfo['coord'] for data_sample in data_samples
        ]).to(batch_inputs)
        cell = torch.stack([
            data_sample.metainfo['cell'] for data_sample in data_samples
        ]).to(batch_inputs)

        feats = self.generator(batch_inputs, coord, cell, **kwargs)

        return feats

    def forward_inference(self, batch_inputs, data_samples=None, **kwargs):
        """Forward inference.
            Returns predictions of validation, testing, and simple inference.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            List[EditDataSample]: predictions.
        """

        feats = self.forward_tensor(batch_inputs, data_samples, test_mode=True)
        feats = self.data_preprocessor.destructor(feats)

        # reshape for eval
        ih, iw = batch_inputs.shape[-2:]
        coord_count = data_samples[0].metainfo['coord'].shape[0]
        s = math.sqrt(coord_count / (ih * iw))
        shape = [len(data_samples), round(ih * s), round(iw * s), 3]
        feats = feats.view(shape).permute(0, 3, 1, 2).contiguous().to('cpu')

        predictions = []
        for idx in range(feats.shape[0]):
            predictions.append(
                EditDataSample(
                    pred_img=PixelData(data=feats[idx]),
                    metainfo=data_samples[idx].metainfo))

        return predictions
