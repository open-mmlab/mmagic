# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch

from mmagic.models.base_models import BaseEditModel
from mmagic.registry import MODELS
from mmagic.structures import DataSample


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

    def forward_tensor(self, inputs, data_samples=None, **kwargs):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        coord = torch.stack(data_samples.metainfo['coord']).to(inputs)
        cell = torch.stack(data_samples.metainfo['cell']).to(inputs)

        feats = self.generator(inputs, coord, cell, **kwargs)

        return feats

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (BaseDataElement, optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            List[DataSample]: predictions.
        """
        # NOTE: feats: shape [bz, N, 3]
        feats = self.forward_tensor(inputs, data_samples, test_mode=True)

        # reshape for eval, [bz, N, 3] -> [bz, 3, H, W]
        ih, iw = inputs.shape[-2:]
        # metainfo in stacked data sample is a list, fetch by indexing
        coord_count = data_samples.metainfo['coord'][0].shape[0]
        s = math.sqrt(coord_count / (ih * iw))
        shape = [len(data_samples), round(ih * s), round(iw * s), 3]
        feats = feats.view(shape).permute(0, 3, 1, 2).contiguous()

        feats = self.data_preprocessor.destruct(feats, data_samples)

        predictions = DataSample(pred_img=feats.cpu())

        return predictions
