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


class InpaintingInferencer(BaseMMEditInferencer):

    func_kwargs = dict(
        preprocess=['img', 'mask'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=['print_result', 'pred_out_file', 'get_datasample'])

    def preprocess(self, img: InputsType, mask: InputsType) -> Dict:
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
                data['data_samples'][0].mask.data, [self.device])[0]

        return data

    def forward(self, inputs: InputsType) -> PredType:
        with torch.no_grad():
            result, x = self.model(mode='tensor', **inputs)
        return result

    def visualize(self,
                  preds: PredType,
                  data: Dict = None,
                  result_out_dir: str = '') -> List[np.ndarray]:
        result = preds[0]
        masks = data['data_samples'][0].mask.data
        masked_imgs = data['inputs'][0]
        result = result * masks + masked_imgs * (1. - masks)

        result = tensor2img(result)[..., ::-1]
        mmcv.imwrite(result, result_out_dir)

        return result
