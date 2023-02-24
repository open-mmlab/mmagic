# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
# from .inferencer_utils import (calculate_grid_size, colorization_inference,
#                                   delete_cfg, init_model,
#                                   inpainting_inference,
#                                   matting_inference,
#                                   restoration_face_inference,
#                                   restoration_inference,
#                                   restoration_video_inference,
#                                   sample_conditional_model,
#                                   sample_img2img_model,
#                                   sample_unconditional_model,
#                                   set_random_seed,
#                                   video_interpolation_inference)

from .diffusers_wrapped import StableDiffusionInferencer

__all__ = ['StableDiffusionInferencer']

# __all__ = [
#     'init_model', 'delete_cfg', 'set_random_seed',
#     'matting_inference', 'inpainting_inference', 'restoration_inference',
#     'restoration_video_inference', 'restoration_face_inference',
#     'video_interpolation_inference', 'sample_conditional_model',
#     'sample_unconditional_model', 'sample_img2img_model',
#     'colorization_inference', 'calculate_grid_size'
# ]
