# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from mmengine import Config
from torch import Tensor

from mmagic.registry import MODELS
from ..singan import SinGAN

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


@MODELS.register_module()
class PESinGAN(SinGAN):
    """Positional Encoding in SinGAN.

    This modified SinGAN is used to reimplement the experiments in: Positional
    Encoding as Spatial Inductive Bias in GANs, CVPR2021.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType],
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 num_scales: Optional[int] = None,
                 fixed_noise_with_pad: bool = False,
                 first_fixed_noises_ch: int = 1,
                 iters_per_scale: int = 200,
                 noise_weight_init: int = 0.1,
                 lr_scheduler_args: Optional[dict] = None,
                 test_pkl_data: Optional[str] = None,
                 ema_confg: Optional[dict] = None):
        super().__init__(generator, discriminator, data_preprocessor,
                         generator_steps, discriminator_steps, num_scales,
                         iters_per_scale, noise_weight_init, lr_scheduler_args,
                         test_pkl_data, ema_confg)
        self.fixed_noise_with_pad = fixed_noise_with_pad
        self.first_fixed_noises_ch = first_fixed_noises_ch

    def construct_fixed_noises(self):
        """Construct the fixed noises list used in SinGAN."""
        for i, real in enumerate(self.reals):
            h, w = real.shape[-2:]
            if self.fixed_noise_with_pad:
                pad_ = self.get_module(self.generator, 'pad_head')
                h += 2 * pad_
                w += 2 * pad_
            if i == 0:
                noise = torch.randn(1, self.first_fixed_noises_ch, h,
                                    w).to(real)
                self.fixed_noises.append(noise)
            else:
                noise = torch.zeros((1, 1, h, w)).to(real)
                self.fixed_noises.append(noise)
