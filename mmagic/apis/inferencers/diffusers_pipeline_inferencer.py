# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import numpy as np
from mmengine import mkdir_or_exist
from PIL.Image import Image
from torchvision.utils import save_image

from .base_mmagic_inferencer import BaseMMagicInferencer, InputsType, PredType


class DiffusersPipelineInferencer(BaseMMagicInferencer):
    """inferencer that predicts with text2image models."""

    func_kwargs = dict(
        preprocess=['text', 'negative_prompt'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    extra_parameters = dict(height=None, width=None)

    def preprocess(self,
                   text: InputsType,
                   negative_prompt: InputsType = None) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            text(InputsType): text input for text-to-image model.
            negative_prompt(InputsType): negative prompt.

        Returns:
            result(Dict): Results of preprocess.
        """
        result = self.extra_parameters
        result['prompt'] = text

        if negative_prompt:
            result['negative_prompt'] = negative_prompt

        return result

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        images = self.model(**inputs).images

        return images

    def visualize(self,
                  preds: PredType,
                  result_out_dir: str = None) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            preds (List[Union[str, np.ndarray]]): Forward results
                by the inferencer.
            result_out_dir (str): Output directory of image.
                Defaults to ''.

        Returns:
            List[np.ndarray]: Result of visualize
        """
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            if type(preds) is list:
                preds = preds[0]
            if type(preds) is Image:
                preds.save(result_out_dir)
            else:
                save_image(preds, result_out_dir, normalize=True)

        return preds
