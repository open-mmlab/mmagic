# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import mmcv
import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter

from mmedit.utils import tensor2img
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class InpaintingInferencer(BaseMMEditInferencer):
    """inferencer that predicts with inpainting models."""

    func_kwargs = dict(
        preprocess=['img', 'mask'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    def preprocess(self, img: InputsType, mask: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            img(InputsType): Image to be inpainted by models.
            mask(InputsType): Image mask for inpainting models.

        Returns:
            results(Dict): Results of preprocess.
        """
        infer_pipeline_cfg = [
            dict(type='LoadImageFromFile', key='gt', channel_order='bgr'),
            dict(
                type='LoadMask',
                mask_mode='file',
            ),
            dict(type='GetMaskedImage'),
            dict(type='PackEditInputs'),
        ]

        infer_pipeline = Compose(infer_pipeline_cfg)

        # prepare data
        _data = infer_pipeline(dict(gt_path=img, mask_path=mask))
        data = dict()
        data['inputs'] = _data['inputs'] / 255.0
        data = collate([data])
        data['data_samples'] = [_data['data_samples']]
        if 'cuda' in str(self.device):
            data = scatter(data, [self.device])[0]
            data['data_samples'][0].mask.data = scatter(
                data['data_samples'][0].mask.data, [self.device])[0] / 255.0

        # save masks and masked_imgs to visualize
        self.masks = data['data_samples'][0].mask.data * 255
        self.masked_imgs = data['inputs'][0]

        return data

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        with torch.no_grad():
            result, x = self.model(mode='tensor', **inputs)
        return result

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = None) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            preds (List[Union[str, np.ndarray]]): Forward results
                by the inferencer.
            data (List[Dict]): Mask of input image.
            result_out_dir (str): Output directory of image.
                Defaults to ''.

        Returns:
            List[np.ndarray]: Result of visualize
        """
        result = preds[0]
        result = result * self.masks + self.masked_imgs * (1. - self.masks)

        result = tensor2img(result)[..., ::-1]
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            mmcv.imwrite(result, result_out_dir)

        return result
