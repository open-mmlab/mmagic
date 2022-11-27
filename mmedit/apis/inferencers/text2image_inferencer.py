# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import numpy as np
from mmengine import mkdir_or_exist
from torchvision.utils import save_image

from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType


class Text2ImageInferencer(BaseMMEditInferencer):
    """inferencer that predicts with text2image models."""

    func_kwargs = dict(
        preprocess=['text'],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=[])

    extra_parameters = dict(
        width=1280,
        height=768,
        show_progress=True,
        num_inference_steps=250,
        eta=0.8,
        seed=2022)

    def preprocess(self, text: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            img(InputsType): Image to be restored by models.

        Returns:
            data(Dict): Results of preprocess.
        """
        result = self.extra_parameters
        result['text_prompts'] = text

        return result

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        image = self.model.infer(**inputs)['samples']

        return image

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
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            save_image(preds, result_out_dir, normalize=True)

        return preds
