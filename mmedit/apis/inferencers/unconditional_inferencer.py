# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import numpy as np
from typing import Dict, List
from torchvision import utils
from mmengine import mkdir_or_exist

from mmedit.structures import EditDataSample
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class UnconditionalInferencer(BaseMMEditInferencer):

    func_kwargs = dict(
        preprocess=[],
        forward=[],
        visualize=['img_out_dir'],
        postprocess=['print_result', 'pred_out_file', 'get_datasample'])

    def preprocess(self) -> Dict:

        # set model with infer_cfg if it exist else set default value
        if 'infer_cfg' in self.cfg and 'sample_nums' in self.cfg.infer_cfg:
            sample_nums = self.cfg.infer_cfg.sample_nums
        else:
            sample_nums = 4
        if 'infer_cfg' in self.cfg and 'sample_model' in self.cfg.infer_cfg:
            sample_model = self.cfg.infer_cfg.sample_model
        else:
            sample_model = 'ema'

        preprocess_res = dict(num_batches=sample_nums, sample_model=sample_model)

        return preprocess_res

    def forward(self, inputs: InputsType) -> PredType:
        return self.model(inputs)
    
    def visualize(self,
                preds: PredType,
                data: Dict = None,
                img_out_dir: str = '') -> List[np.ndarray]:
        
        res_list = []
        res_list.extend([item.fake_img.data.cpu() for item in preds])
        results = torch.stack(res_list, dim=0)
        results = (results[:, [2, 1, 0]] + 1.) / 2.

        # save images
        mkdir_or_exist(os.path.dirname(img_out_dir))
        utils.save_image(results, img_out_dir)

        return results

    def _pred2dict(self, data_sample: EditDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (EditDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['fake_img'] = data_sample.fake_img.data.cpu()
        result['noise'] = data_sample.noise.data.cpu()
        return result
