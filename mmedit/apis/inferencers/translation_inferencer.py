# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import numpy as np
from typing import Dict, List
from torchvision import utils
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate

from mmedit.models.base_models import BaseTranslationModel
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class TranslationInferencer(BaseMMEditInferencer):

    func_kwargs = dict(
        preprocess=['img'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=['print_result', 'pred_out_file', 'get_datasample'])

    def preprocess(self, img: InputsType) -> Dict:
        
        assert isinstance(self.model, BaseTranslationModel)

        # get source domain and target domain
        self.target_domain = self.model._default_domain
        source_domain = self.model.get_other_domains(self.target_domain)[0]

        cfg = self.model.cfg
        # build the data pipeline
        test_pipeline = Compose(cfg.test_pipeline)

        # prepare data
        data = dict()
        # dirty code to deal with test data pipeline
        data['pair_path'] = img
        data[f'img_{source_domain}_path'] = img
        data[f'img_{self.target_domain}_path'] = img

        data = collate([test_pipeline(data)])
        data = self.model.data_preprocessor(data, False)
        inputs_dict = data['inputs']

        source_image = inputs_dict[f'img_{source_domain}']
        return source_image

    def forward(self, inputs: InputsType) -> PredType:
        with torch.no_grad():
            results = self.model(
                inputs,
                test_mode=True,
                target_domain=self.target_domain)
        output = results['target']
        return output
    
    def visualize(self,
                preds: PredType,
                data: Dict = None,
                result_out_dir: str = '') -> List[np.ndarray]:
        
        results = (preds[:, [2, 1, 0]] + 1.) / 2.

        # save images
        mkdir_or_exist(os.path.dirname(result_out_dir))
        utils.save_image(results, result_out_dir)

        return results

