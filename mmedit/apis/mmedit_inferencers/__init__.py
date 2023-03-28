# Copyright (c) OpenMMLab. All rights reserved.
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
    'VideoInterpolationInferencer', 'VideoRestorationInferencer'
]
