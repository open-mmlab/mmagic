# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import mmcv
import numpy as np
import torch
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter

from mmedit.utils import tensor2img
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class RestorationInferencer(BaseMMEditInferencer):

    func_kwargs = dict(
        preprocess=['img'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=['print_result', 'pred_out_file', 'get_datasample'])

    def preprocess(self, img: InputsType, ref: InputsType = None) -> Dict:

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
        data['inputs'] = _data['inputs'] / 255.0
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
        with torch.no_grad():
            result = self.model(mode='tensor', **inputs)
        return result

    def visualize(self,
                  preds: PredType,
                  data: Dict = None,
                  result_out_dir: str = '') -> List[np.ndarray]:

        output = tensor2img(preds[0])
        mmcv.imwrite(output, result_out_dir)

        return output
