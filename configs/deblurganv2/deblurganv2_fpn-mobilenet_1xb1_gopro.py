# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../_base_/default_runtime.py']

save_dir = './work_dir/'

model = dict(
    type='DeblurGanV2',
    generator=dict(
        type='DeblurGanV2Generator',
        backbone='FPNMobileNet',
        norm_layer='instance',
        output_ch=3,
        num_filter=64,
        num_filter_fpn=128,
    ),
    discriminator=dict(
        type='DeblurGanV2Discriminator',
        backbone='DoubleGan',
        norm_layer='instance',
        d_layers=3,
    ),
    pixel_loss=dict(
        type='PerceptualLoss', layer_weights={'14': 1}, criterion='mse'),
    disc_loss=dict(type='AdvLoss', loss_type='ragan-ls'),
    adv_lambda=0.001,
    warmup_num=3,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5] * 3,
        std=[127.5] * 3,
    ))

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='PairedAlbuTransForms', size=256, lq_key='img', gt_key='gt'),
    dict(
        type='AlbuCorruptFunction',
        keys=['img'],
        config=[{
            'name': 'cutout',
            'prob': 0.5,
            'num_holes': 3,
            'max_h_size': 25,
            'max_w_size': 25
        }, {
            'name': 'jpeg',
            'quality_lower': 70,
            'quality_upper': 90
        }, {
            'name': 'motion_blur'
        }, {
            'name': 'median_blur'
        }, {
            'name': 'gamma'
        }, {
            'name': 'rgb_shift'
        }, {
            'name': 'hsv_shift'
        }, {
            'name': 'sharpen'
        }]),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='PairedAlbuTransForms', size=256, lq_key='img', gt_key='gt'),
    dict(
        type='AlbuCorruptFunction',
        keys=['img'],
        config=[{
            'name': 'cutout',
            'prob': 0.5,
            'num_holes': 3,
            'max_h_size': 25,
            'max_w_size': 25
        }, {
            'name': 'jpeg',
            'quality_lower': 70,
            'quality_upper': 90
        }, {
            'name': 'motion_blur'
        }, {
            'name': 'median_blur'
        }, {
            'name': 'gamma'
        }, {
            'name': 'rgb_shift'
        }, {
            'name': 'hsv_shift'
        }, {
            'name': 'sharpen'
        }]),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='PackInputs')
]

data_root = 'data/gopro'

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root=data_root + '/train',
        data_prefix=dict(img='input', gt='target'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root=data_root + '/test',
        data_prefix=dict(img='input', gt='target'),
        pipeline=val_pipeline))

test_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root=data_root + '/test',
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999))),
)

# learning policy
param_scheduler = dict(
    type='LinearLR',
    start_factor=0.0001,
    end_factor=0.0000001,
    begin=50,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
        save_best='auto',
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

load_from = 'https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/'\
            'weight/DeblurGANv2_fpn-mobilenet.pth'
