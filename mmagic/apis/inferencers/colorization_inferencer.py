# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import mmcv
import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate

from mmagic.structures import DataSample
from mmagic.utils import tensor2img
from .base_mmagic_inferencer import BaseMMagicInferencer, InputsType, PredType


class ColorizationInferencer(BaseMMagicInferencer):
    """inferencer that predicts with colorization models."""

    func_kwargs = dict(
        preprocess=['img'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    def preprocess(self, img: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            img(InputsType): Image to be translated by models.

        Returns:
            results(Dict): Results of preprocess.
        """
        # build the data pipeline
        test_pipeline = Compose(self.model.cfg.test_pipeline)
        # prepare data
        data = dict(img_path=img)
        _data = test_pipeline(data)
        data = dict()
        data['inputs'] = _data['inputs'] / 255.0
        data = collate([data])
        data['data_samples'] = [_data['data_samples']]
        if 'empty_box' not in data['data_samples'][0]:
            data['data_samples'][0].set_data({'empty_box': True})
        if not data['data_samples'][0].empty_box:
            data['data_samples'][0].cropped_img.data = \
                data['data_samples'][0].cropped_img.data / 255.0
        if 'cuda' in str(self.device):
            data['inputs'] = data['inputs'].cuda()
            data['data_samples'][0] = data['data_samples'][0].cuda()
        data['data_samples'] = DataSample.stack(data['data_samples'])
        return data

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        with torch.no_grad():
            result = self.model(mode='tensor', **inputs)
        return result

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = None) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            preds (List[Union[str, np.ndarray]]): Forward results
                by the inferencer.
            data (List[Dict]): Not needed by this kind of inferencer.
            result_out_dir (str): Output directory of image.
                Defaults to ''.

        Returns:
            List[np.ndarray]: Result of visualize
        """
        results = tensor2img(preds[0])
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            mmcv.imwrite(results, result_out_dir)

        return results
