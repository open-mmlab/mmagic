# Copyright (c) OpenMMLab. All rights reserved.
from .inference import delete_cfg, init_model
from .inpainting_inference import inpainting_inference
from .matting_inference import matting_inference
from .restoration_face_inference import restoration_face_inference
from .restoration_inference import restoration_inference
from .restoration_video_inference import restoration_video_inference
from .video_interpolation_inference import video_interpolation_inference

__all__ = [
    'init_model',
    'delete_cfg',
    'matting_inference',
    'inpainting_inference',
    'restoration_inference',
    'restoration_video_inference',
    'restoration_face_inference',
    'video_interpolation_inference',
]
