# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from torch.optim import Adam

from mmagic.engine import VisualizationHook
from mmagic.evaluation import (FrechetInceptionDistance, PerceptualPathLength,
                               PrecisionAndRecall)
from mmagic.models import BaseGAN

with read_base():
    from .._base_.datasets.lsun_stylegan import *  # noqa: F403,F405
    from .._base_.gen_default_runtime import *  # noqa: F403,F405
    from .._base_.models.base_styleganv2 import *  # noqa: F403,F405

# reg params
d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

ema_half_life = 10.  # G_smoothing_kimg

model.update(
    generator=dict(out_size=256),
    discriminator=dict(in_size=256),
    ema_config=dict(
        type=ExponentialMovingAverage,
        interval=1,
        momentum=1. - (0.5**(32. / (ema_half_life * 1000.)))),
    loss_config=dict(
        r1_loss_weight=10. / 2. * d_reg_interval,
        r1_interval=d_reg_interval,
        norm_mode='HWC',
        g_reg_interval=g_reg_interval,
        g_reg_weight=2. * g_reg_interval,
        pl_batch_shrink=2))

train_cfg.update(max_iters=800002)

optim_wrapper.update(
    generator=dict(
        optimizer=dict(
            type=Adam, lr=0.002 * g_reg_ratio, betas=(0, 0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type=Adam, lr=0.002 * d_reg_ratio, betas=(0, 0.99**d_reg_ratio))))

batch_size = 4
data_root = './data/lsun-cat'

train_dataloader.update(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader.update(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader.update(
    batch_size=batch_size, dataset=dict(data_root=data_root))

# VIS_HOOK
custom_hooks = [
    dict(
        type=VisualizationHook,
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type=BaseGAN, name='fake_img'))
]

# METRICS
metrics = [
    dict(
        type=FrechetInceptionDistance,
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(type=PrecisionAndRecall, fake_nums=50000, prefix='PR-50K'),
    dict(type=PerceptualPathLength, fake_nums=50000, prefix='ppl-w')
]
# NOTE: config for save multi best checkpoints
# default_hooks.update(
#     checkpoint=dict(
#         save_best=['FID-Full-50k/fid', 'IS-50k/is'],
#         rule=['less', 'greater']))
default_hooks.update(checkpoint=dict(save_best='FID-Full-50k/fid'))

val_evaluator.update(metrics=metrics)
test_evaluator.update(metrics=metrics)
