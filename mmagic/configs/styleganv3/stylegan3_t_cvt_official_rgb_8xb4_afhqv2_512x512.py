# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.unconditional_imgs_flip_512x512 import *
    from .._base_.gen_default_runtime import *
    from .._base_.models.base_styleganv3 import *

from mmagic.evaluation.metrics.fid import FrechetInceptionDistance
from mmagic.models.editors.stylegan3.stylegan3_modules import SynthesisNetwork

synthesis_cfg = {
    'type': SynthesisNetwork,
    'channel_base': 32768,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}
model.update(
    generator=dict(
        out_size=512,
        img_channels=3,
        rgb2bgr=True,
        synthesis_cfg=synthesis_cfg),
    discriminator=dict(in_size=512))

batch_size = 4
data_root = 'data/afhqv2/'

train_dataloader.update(
    batch_size=batch_size, dataset=dict(data_root=data_root))
val_dataloader.update(batch_size=batch_size, dataset=dict(data_root=data_root))
test_dataloader.update(
    batch_size=batch_size, dataset=dict(data_root=data_root))

train_cfg = train_dataloader = optim_wrapper = None

metrics = [
    dict(
        type=FrechetInceptionDistance,
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
# NOTE: config for save multi best checkpoints
# default_hooks = dict(
#     checkpoint=dict(
#         save_best=['FID-Full-50k/fid', 'IS-50k/is'],
#         rule=['less', 'greater']))

default_hooks.update(checkpoint=dict(save_best='FID-Full-50k/fid'))
val_evaluator.update(metrics=metrics)
test_evaluator.update(metrics=metrics)
