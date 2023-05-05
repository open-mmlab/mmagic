# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from torchvision import utils

from mmagic.models.base_models import BaseTranslationModel
from .base_mmagic_inferencer import BaseMMagicInferencer, InputsType, PredType


class TranslationInferencer(BaseMMagicInferencer):
    """inferencer that predicts with translation models."""

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
        assert isinstance(self.model, BaseTranslationModel)

        # get source domain and target domain
        self.target_domain = self.model._default_domain
        source_domain = self.model.get_other_domains(self.target_domain)[0]

        cfg = self.model.cfg
        # build the data pipeline
        test_pipeline = Compose(cfg.test_pipeline)

        # prepare data
        # dirty code to deal with test data pipeline
        data = dict()
        data['pair_path'] = img
        data['img_A_path'] = img
        data['img_B_path'] = img
        data = collate([test_pipeline(data)])
        data = self.model.data_preprocessor(data, False)

        inputs_dict = data['inputs']
        results = inputs_dict[f'img_{source_domain}']
        return results

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        with torch.no_grad():
            results = self.model(
                inputs, test_mode=True, target_domain=self.target_domain)
        output = results['target']
        return output

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
        results = (preds[:, [2, 1, 0]] + 1.) / 2.

        # save images
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            utils.save_image(results, result_out_dir)

        return results
