# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

from mmagic.evaluation.metrics import FrechetInceptionDistance
from mmagic.models.data_preprocessors import DataPreprocessor
from mmagic.models.editors.biggan import (BigGAN, BigGANDiscriminator,
                                          BigGANGenerator)

with read_base():
    from .._base_.datasets.imagenet_noaug_128 import *
    from .._base_.gen_default_runtime import *

# setting image size to 512x512
train_dataloader.dataset.pipeline[2].scale = (512, 512)
test_dataloader.dataset.pipeline[2].scale = (512, 512)
val_dataloader.dataset.pipeline[2].scale = (512, 512)

ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.0001,
    update_buffers=True,
    start_iter=20000)

model = dict(
    type=BigGAN,
    num_classes=1000,
    data_preprocessor=dict(type=DataPreprocessor),
    ema_config=ema_config,
    generator=dict(
        type=BigGANGenerator,
        output_scale=512,
        noise_size=128,
        num_classes=1000,
        base_channels=128,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        sn_style='torch',
        act_cfg=dict(type='ReLU', inplace=True),
        concat_noise=True,
        auto_sync_bn=False,
        rgb2bgr=True,
        init_cfg=dict(type='ortho')),
    discriminator=dict(
        type=BigGANDiscriminator,
        input_scale=512,
        num_classes=1000,
        base_channels=128,
        sn_eps=1e-6,
        sn_style='torch',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True,
        init_cfg=dict(type='ortho')))

train_cfg = train_dataloader = optim_wrapper = None

metrics = [
    dict(
        type=FrechetInceptionDistance,
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(
        type='IS',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
