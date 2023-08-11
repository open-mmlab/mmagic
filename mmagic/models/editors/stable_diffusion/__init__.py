# Copyright (c) OpenMMLab. All rights reserved.
from .stable_diffusion import StableDiffusion
from .stable_diffusion_inpaint import StableDiffusionInpaint
from .vae import AutoencoderKL

__all__ = ['StableDiffusion', 'AutoencoderKL', 'StableDiffusionInpaint']
