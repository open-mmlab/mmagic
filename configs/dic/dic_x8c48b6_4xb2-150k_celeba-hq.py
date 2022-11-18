_base_ = '../_base_/default_runtime.py'

experiment_name = 'dic_x8c48b6_4xb2-150k_celeba-hq'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 8

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
model = dict(
    type='DIC',
    generator=dict(
        type='DICNet', in_channels=3, out_channels=3, mid_channels=48),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[129.795, 108.12, 96.39],
        std=[255, 255, 255],
    ))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='Resize',
        scale=(128, 128),
        keys=['gt'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=1 / 8,
        keep_ratio=True,
        keys=['gt'],
        output_keys=['img'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='GenerateFacialHeatmap',
        image_key='gt',
        ori_size=128,
        target_size=32,
        sigma=1.0),
    dict(type='PackEditInputs')
]
valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='Resize',
        scale=(128, 128),
        keys=['gt'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=1 / 8,
        keep_ratio=True,
        keys=['gt'],
        output_keys=['img'],
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackEditInputs')
]
test_pipeline = valid_pipeline

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=4,
    batch_size=2,  # gpus 4
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='celeba', task_name='fsr'),
        data_root=data_root + '/CelebA-HQ',
        data_prefix=dict(gt='train_256/all_256'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='celeba', task_name='fsr'),
        data_root=data_root + '/CelebA-HQ',
        data_prefix=dict(gt='test_256/all_256'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR', crop_border=scale),
    dict(type='SSIM', crop_border=scale),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=150_000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-4)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    milestones=[10000, 20000, 40000, 80000],
    gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
