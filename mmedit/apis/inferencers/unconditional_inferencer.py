# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import numpy as np
import torch
from mmengine import mkdir_or_exist
from torchvision import utils

from mmedit.structures import EditDataSample
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class UnconditionalInferencer(BaseMMEditInferencer):
    """inferencer that predicts with unconditional models."""

    func_kwargs = dict(
        preprocess=[],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    extra_parameters = dict(num_batches=4, sample_model='ema')

    def preprocess(self) -> Dict:
        """Process the inputs into a model-feedable format.

        Returns:
            results(Dict): Results of preprocess.
        """
        num_batches = self.extra_parameters['num_batches']
        sample_model = self.extra_parameters['sample_model']

        results = dict(num_batches=num_batches, sample_model=sample_model)

        return results

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        return self.model(inputs)

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = '') -> List[np.ndarray]:
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
        res_list = []
        res_list.extend([item.fake_img.data.cpu() for item in preds])
        results = torch.stack(res_list, dim=0)
        results = (results[:, [2, 1, 0]] + 1.) / 2.

        # save images
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            utils.save_image(results, result_out_dir)

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
