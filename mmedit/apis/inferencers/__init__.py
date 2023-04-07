# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch

from mmedit.utils import ConfigType

from .colorization_inferencer import ColorizationInferencer
from .conditional_inferencer import ConditionalInferencer
from .eg3d_inferencer import EG3DInferencer
from .image_super_resolution_inferencer import ImageSuperResolutionInferencer
# yapf: disable
from .inference_functions import (calculate_grid_size, colorization_inference,
                                  delete_cfg, init_model, inpainting_inference,
                                  matting_inference,
                                  restoration_face_inference,
                                  restoration_inference,
                                  restoration_video_inference,
                                  sample_conditional_model,
                                  sample_img2img_model,
                                  sample_unconditional_model, set_random_seed,
                                  video_interpolation_inference)
# yapf: enable
from .inpainting_inferencer import InpaintingInferencer
from .matting_inferencer import MattingInferencer
from .text2image_inferencer import Text2ImageInferencer
from .translation_inferencer import TranslationInferencer
from .unconditional_inferencer import UnconditionalInferencer
from .video_interpolation_inferencer import VideoInterpolationInferencer
from .video_restoration_inferencer import VideoRestorationInferencer
from .controlnet_animation_inferencer import ControlnetAnimationInferencer

__all__ = [
    'init_model', 'delete_cfg', 'set_random_seed', 'matting_inference',
    'inpainting_inference', 'restoration_inference',
    'restoration_video_inference', 'restoration_face_inference',
    'video_interpolation_inference', 'sample_conditional_model',
    'sample_unconditional_model', 'sample_img2img_model',
    'colorization_inference', 'calculate_grid_size', 'ColorizationInferencer',
    'ConditionalInferencer', 'EG3DInferencer', 'InpaintingInferencer',
    'MattingInferencer', 'ImageSuperResolutionInferencer',
    'Text2ImageInferencer', 'TranslationInferencer', 'UnconditionalInferencer',
    'VideoInterpolationInferencer', 'VideoRestorationInferencer',
    'ControlnetAnimationInferencer'
]


class Inferencers:
    """Class to assign task to different inferencers.

    Args:
        task (str): Inferencer task.
        config (str or ConfigType): Model config or the path to it.
        ckpt (str, optional): Path to the checkpoint.
        device (str, optional): Device to run inference. If None, the best
            device will be automatically used.
        seed (int): The random seed used in inference. Defaults to 2022.
    """

    def __init__(self,
                 task: Optional[str] = None,
                 config: Optional[Union[ConfigType, str]] = None,
                 ckpt: Optional[str] = None,
                 device: torch.device = None,
                 extra_parameters: Optional[Dict] = None,
                 seed: int = 2022) -> None:
        self.task = task
        if self.task in ['conditional', 'Conditional GANs']:
            self.inferencer = ConditionalInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['colorization', 'Colorization']:
            self.inferencer = ColorizationInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['unconditional', 'Unconditional GANs']:
            self.inferencer = UnconditionalInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['matting', 'Matting']:
            self.inferencer = MattingInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['inpainting', 'Inpainting']:
            self.inferencer = InpaintingInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['translation', 'Image2Image']:
            self.inferencer = TranslationInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['Image super-resolution', 'Image Super-Resolution']:
            self.inferencer = ImageSuperResolutionInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['video_restoration', 'Video Super-Resolution']:
            self.inferencer = VideoRestorationInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['video_interpolation', 'Video Interpolation']:
            self.inferencer = VideoInterpolationInferencer(
                config, ckpt, device, extra_parameters)
        elif self.task in [
                'text2image', 'Text2Image', 'Text2Image, Image2Image'
        ]:
            self.inferencer = Text2ImageInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['3D_aware_generation', '3D-aware Generation']:
            self.inferencer = EG3DInferencer(
                config, ckpt, device, extra_parameters, seed=seed)
        elif self.task in ['controlnet_animation']:
            self.inferencer = ControlnetAnimationInferencer(config)
        else:
            raise ValueError(f'Unknown inferencer task: {self.task}')

    def __call__(self, **kwargs) -> Union[Dict, List[Dict]]:
        """Call the inferencer.

        Args:
            kwargs: Keyword arguments for the inferencer.

        Returns:
            Union[Dict, List[Dict]]: Results of inference pipeline.
        """
        return self.inferencer(**kwargs)

    def get_extra_parameters(self) -> List[str]:
        """Each inferencer may has its own parameters. Call this function to
        get these parameters.

        Returns:
            List[str]: List of unique parameters.
        """
        return self.inferencer.get_extra_parameters()
