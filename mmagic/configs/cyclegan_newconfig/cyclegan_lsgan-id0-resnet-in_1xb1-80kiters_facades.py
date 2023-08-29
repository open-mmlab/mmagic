# Copyright (c) OpenMMLab. All rights reserved.
from mmcv import KeyMapper
from mmengine.config import read_base
from torch.optim import Adam

from mmagic.datasets.transforms.formatting import PackInputs
from mmagic.engine.hooks import VisualizationHook
from mmagic.engine.schedulers.linear_lr_scheduler_with_interval import \
    LinearLrInterval
from mmagic.evaluation.metrics import TransFID, TransIS

with read_base():
    from .._base_.datasets.unpaired_imgs_256x256 import * # noqa: F401,F403
    from .._base_.gen_default_runtime import *  # noqa: F401,F403
    from .._base_.models.cyclegan.base_cyclegan import *  # noqa: F401,F403

train_cfg.update(dict(max_iters=80000))
domain_a = 'photo'
domain_b = 'mask'
model.update(
    dict(
        loss_config=dict(cycle_loss_weight=10., id_loss_weight=0.),
        default_domain=domain_a,
        reachable_domains=[domain_a, domain_b],
        related_domains=[domain_a, domain_b],
        data_preprocessor=dict(
            data_keys=[f'img_{domain_a}', f'img_{domain_b}'])))

param_scheduler = dict(
    type=LinearLrInterval,
    interval=400,
    by_epoch=False,
    start_factor=0.0002,
    end_factor=0,
    begin=40000,
    end=80000)
# dataroot = './data/cyclegan/facades'
dataroot = './data/cyclegan/horse2zebra'

val_pipeline = test_pipeline
key_mapping = dict(
    type=KeyMapper,
    mapping={
        f'img_{domain_a}': 'img_A',
        f'img_{domain_b}': 'img_B'
    },
    remapping={
        f'img_{domain_a}': f'img_{domain_a}',
        f'img_{domain_b}': f'img_{domain_b}'
    })
pack_input = dict(
    type=PackInputs,
    keys=[f'img_{domain_a}', f'img_{domain_b}'],
    data_keys=[f'img_{domain_a}', f'img_{domain_b}'])

train_pipeline += [key_mapping, pack_input]
val_pipeline += [key_mapping, pack_input]
test_pipeline += [key_mapping, pack_input]
train_dataloader.update(dict(dataset=dict(data_root=dataroot)))
val_dataloader.update(dict(dataset=dict(data_root=dataroot, test_mode=True)))
test_dataloader = val_dataloader
optim_wrapper.update(
    dict(
        generators=dict(
            optimizer=dict(type=Adam, lr=0.0002, betas=(0.5, 0.999))),
        discriminators=dict(
            optimizer=dict(type=Adam, lr=0.0002, betas=(0.5, 0.999)))))

custom_hooks = [
    dict(
        type=VisualizationHook,
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=[
            dict(type='Translation', name='trans'),
            dict(type='TranslationVal', name='trans_val')
        ])
]

num_images = 106

metrics = [
    dict(
        type=TransIS,
        prefix='IS-Full',
        fake_nums=num_images,
        fake_key=f'fake_{domain_a}',
        use_pillow_resize=False,
        resize_method='bilinear',
        inception_style='PyTorch'),
    dict(
        type=TransFID,
        prefix='FID-Full',
        fake_nums=num_images,
        inception_style='PyTorch',
        real_key=f'img_{domain_a}',
        fake_key=f'fake_{domain_a}')
]

val_evaluator.update(dict(metrics=metrics))
test_evaluator.update(dict(metrics=metrics))
