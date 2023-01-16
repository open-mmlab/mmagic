# Copyright (c) OpenMMLab. All rights reserved.
from .gfpgan import GFPGAN
from .gfpgan_generator import GFPGANv1Clean
from .stylegan2_clean import StyleGAN2GeneratorClean

__all__ = ['GFPGANv1Clean', 'StyleGAN2GeneratorClean', 'GFPGAN']
