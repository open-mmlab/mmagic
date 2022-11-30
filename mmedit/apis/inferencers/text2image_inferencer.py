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
        scheduler_kwargs=None,
        height=None,
        width=None,
        init_image=None,
        batch_size=1,
        num_inference_steps=1000,
        skip_steps=0,
        show_progress=False,
        text_prompts=[],
        image_prompts=[],
        eta=0.8,
        clip_guidance_scale=5000,
        init_scale=1000,
        tv_scale=0.,
        sat_scale=0.,
        range_scale=150,
        cut_overview=[12] * 400 + [4] * 600,
        cut_innercut=[4] * 400 + [12] * 600,
        cut_ic_pow=[1] * 1000,
        cut_icgray_p=[0.2] * 400 + [0] * 600,
        cutn_batches=4,
        seed=2022)

    def preprocess(self, text: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            text(InputsType): text input for text-to-image model.

        Returns:
            result(Dict): Results of preprocess.
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
            result_out_dir (str): Output directory of image.
                Defaults to ''.

        Returns:
            List[np.ndarray]: Result of visualize
        """
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            save_image(preds, result_out_dir, normalize=True)

        return preds
