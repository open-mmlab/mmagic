# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import mmcv
import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose

from mmagic.utils import tensor2img
from .base_mmagic_inferencer import BaseMMagicInferencer, InputsType, PredType


class InpaintingInferencer(BaseMMagicInferencer):
    """inferencer that predicts with inpainting models."""

    func_kwargs = dict(
        preprocess=['img', 'mask'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    def _init_pipeline(self, cfg) -> Compose:
        """Initialize the test pipeline."""
        return None

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
            dict(type='PackInputs'),
        ]

        infer_pipeline = Compose(infer_pipeline_cfg)

        # prepare data
        _data = infer_pipeline(dict(gt_path=img, mask_path=mask))
        data = dict()
        data['inputs'] = [_data['inputs']]
        data['data_samples'] = [_data['data_samples']]
        return data

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        inputs = self.model.data_preprocessor(inputs)
        with torch.no_grad():
            result = self.model(mode='predict', **inputs)
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
        result = preds[0].output.pred_img / 255.

        result = tensor2img(result)[..., ::-1]
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            mmcv.imwrite(result, result_out_dir)

        return result
