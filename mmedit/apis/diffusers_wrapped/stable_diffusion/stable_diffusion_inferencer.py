# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

import cv2
import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from mmedit.utils import InputsType, OutputKeys
from ..diffusers_inferencer import DiffusersInferencer


class StableDiffusionInferencer(DiffusersInferencer):
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

    def __init__(self,
                 config: Union[ConfigType, str],
                 ckpt: Optional[str],
                 device: Optional[str] = None,
                 extra_parameters: Optional[Dict] = None,
                 seed: int = 2022,
                 **kwargs) -> None:
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub.
            device: str = 'gpu'
        """
        # Load config to cfg
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # build upon the diffuser stable diffusion pipeline
        self.inferencer = StableDiffusionPipeline.from_pretrained(config)

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

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')

        image = self.pipeline(
            prompt=inputs.get('text'),
            height=inputs.get('height'),
            width=inputs.get('width'),
            num_inference_steps=inputs.get('num_inference_steps', 50),
            guidance_scale=inputs.get('guidance_scale', 7.5),
            negative_prompt=inputs.get('negative_prompt'),
            num_images_per_prompt=inputs.get('num_images_per_prompt', 1),
            eta=inputs.get('eta', 0.0),
            generator=inputs.get('generator'),
            latents=inputs.get('latents'),
            output_type=inputs.get('output_type', 'pil'),
            return_dict=inputs.get('return_dict', True),
            callback=inputs.get('callback'),
            callback_steps=inputs.get('callback_steps', 1)).images[0]
        return image

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}
