# Copyright (c) OpenMMLab. All rights reserved.
# define GAN model
from mmagic.models.base_models.average_model import ExponentialMovingAverage
from mmagic.models.data_preprocessors import DataPreprocessor
from mmagic.models.editors.stylegan2 import StyleGAN2Discriminator
from mmagic.models.editors.stylegan3 import StyleGAN3, StyleGAN3Generator

d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

model = dict(
    type=StyleGAN3,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(
        type=StyleGAN3Generator,  # StyleGANv3Generator
        noise_size=512,
        style_channels=512,
        out_size=None,  # Need to be set.
        img_channels=3,
    ),
    discriminator=dict(
        type=StyleGAN2Discriminator,
        in_size=None,  # Need to be set.
    ),
    ema_config=dict(type=ExponentialMovingAverage),
    loss_config=dict(
        r1_loss_weight=10. / 2. * d_reg_interval,
        r1_interval=d_reg_interval,
        norm_mode='HWC',
        g_reg_interval=g_reg_interval,
        g_reg_weight=2. * g_reg_interval,
        pl_batch_shrink=2))
