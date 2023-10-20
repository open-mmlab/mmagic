# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import ExponentialMovingAverage

from mmagic.models import DataPreprocessor
from mmagic.models.editors import StyleGAN2
from mmagic.models.editors.stylegan2 import (StyleGAN2Discriminator,
                                             StyleGAN2Generator)

# define GAN model

d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

loss_config = dict(
    r1_loss_weight=10. / 2. * d_reg_interval,
    r1_interval=d_reg_interval,
    norm_mode='HWC',
    g_reg_interval=g_reg_interval,
    g_reg_weight=2. * g_reg_interval,
    pl_batch_shrink=2)

model = dict(
    type=StyleGAN2,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(
        type=StyleGAN2Generator,
        out_size=None,  # Need to be set.
        style_channels=512,
    ),
    discriminator=dict(
        type=StyleGAN2Discriminator,
        in_size=None,  # Need to be set.
    ),
    ema_config=dict(type=ExponentialMovingAverage),
    loss_config=loss_config)
