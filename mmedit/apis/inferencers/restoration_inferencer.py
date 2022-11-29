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


class RestorationInferencer(BaseMMEditInferencer):
    """inferencer that predicts with restoration models."""

    func_kwargs = dict(
        preprocess=['img'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    def preprocess(self, img: InputsType, ref: InputsType = None) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            img(InputsType): Image to be restored by models.
            ref(InputsType): Reference image for resoration models.
                Defaults to None.

        Returns:
            data(Dict): Results of preprocess.
        """
        cfg = self.model.cfg
        device = next(self.model.parameters()).device  # model device

        # select the data pipeline
        if cfg.get('demo_pipeline', None):
            test_pipeline = cfg.demo_pipeline
        elif cfg.get('test_pipeline', None):
            test_pipeline = cfg.test_pipeline
        else:
            test_pipeline = cfg.val_pipeline

        # remove gt from test_pipeline
        keys_to_remove = ['gt', 'gt_path']
        for key in keys_to_remove:
            for pipeline in list(test_pipeline):
                if 'key' in pipeline and key == pipeline['key']:
                    test_pipeline.remove(pipeline)
                if 'keys' in pipeline and key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                    if len(pipeline['keys']) == 0:
                        test_pipeline.remove(pipeline)
                if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)
        # build the data pipeline
        test_pipeline = Compose(test_pipeline)
        # prepare data
        if ref:  # Ref-SR
            data = dict(img_path=img, ref_path=ref)
        else:  # SISR
            data = dict(img_path=img)
        _data = test_pipeline(data)
        data = dict()
        data_preprocessor = cfg['model']['data_preprocessor']
        mean = torch.Tensor(data_preprocessor['mean']).view([3, 1, 1])
        std = torch.Tensor(data_preprocessor['std']).view([3, 1, 1])
        data['inputs'] = (_data['inputs'] - mean) / std
        data = collate([data])
        if ref:
            data['data_samples'] = [_data['data_samples']]
        if 'cuda' in str(device):
            data = scatter(data, [device])[0]
            if ref:
                data['data_samples'][0].img_lq.data = data['data_samples'][
                    0].img_lq.data.to(device)
                data['data_samples'][0].ref_lq.data = data['data_samples'][
                    0].ref_lq.data.to(device)
                data['data_samples'][0].ref_img.data = data['data_samples'][
                    0].ref_img.data.to(device)
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
