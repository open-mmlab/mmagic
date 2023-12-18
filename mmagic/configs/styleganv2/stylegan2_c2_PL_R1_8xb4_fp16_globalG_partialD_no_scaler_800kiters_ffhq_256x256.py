# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.optim import AmpOptimWrapper

with read_base():
    from mmagic.configs.styleganv2.stylegan2_c2_8xb4_800kiters_ffhq_256x256 import *

model.update(
    generator=dict(out_size=256, fp16_enabled=True),
    discriminator=dict(in_size=256, fp16_enabled=False, num_fp16_scales=4),
)
train_cfg.update(max_iters=800000)
optim_wrapper.update(
    generator=dict(type=AmpOptimWrapper, loss_scale=512),
    discriminator=dict(type=AmpOptimWrapper, loss_scale=512))
