# Copyright (c) OpenMMLab. All rights reserved.
import torch
import mmcv
import numpy as np
from typing import Dict, List
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torch.nn.parallel import scatter

from mmedit.utils import tensor2img
from mmedit.structures import EditDataSample
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class InpaintingInferencer(BaseMMEditInferencer):

    func_kwargs = dict(
        preprocess=['img', 'mask'],
        forward=[],
        visualize=['img_out_dir'],
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
                img_out_dir: str = '') -> List[np.ndarray]:
        result = preds[0]
        masks = data['data_samples'][0].mask.data
        masked_imgs = data['inputs'][0]
        result = result * masks + masked_imgs * (1. - masks)

        result = tensor2img(result)[..., ::-1]
        mmcv.imwrite(result, img_out_dir)

        return result

    def _pred2dict(self, data_sample: torch.Tensor) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (torch.Tensor): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['infer_res'] = data_sample
        return result
