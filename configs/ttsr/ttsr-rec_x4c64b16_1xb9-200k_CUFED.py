_base_ = '../_base_/default_runtime.py'

experiment_name = 'ttsr-rec_x4c64b16_1xb9-200k_CUFED'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
model = dict(
    type='TTSR',
    generator=dict(
        type='TTSRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=(16, 16, 8, 4)),
    extractor=dict(type='LTE'),
    transformer=dict(type='SearchTransformer'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    ))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(
        type='LoadImageFromFile',
        key='ref',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='ModCrop', key='gt'),
    dict(type='CropLike', target_key='ref', reference_key='gt'),
    dict(
        type='Resize',
        scale=1 / scale,
        keep_ratio=True,
        keys=['gt', 'ref'],
        output_keys=['img', 'ref_down'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=float(scale),
        keep_ratio=True,
        keys=['img', 'ref_down'],
        output_keys=['img_lq', 'ref_lq'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Flip',
        keys=['img', 'gt', 'img_lq'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip',
        keys=['img', 'gt', 'img_lq'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(
        type='RandomTransposeHW',
        keys=['img', 'gt', 'img_lq'],
        transpose_ratio=0.5),
    dict(
        type='Flip',
        keys=['ref', 'ref_lq'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip',
        keys=['ref', 'ref_lq'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(
        type='RandomTransposeHW', keys=['ref', 'ref_lq'], transpose_ratio=0.5),
    dict(type='PackEditInputs')
]
valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(
        type='LoadImageFromFile',
        key='ref',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='ModCrop', key='gt'),
    dict(type='CropLike', target_key='ref', reference_key='gt'),
    dict(
        type='Resize',
        scale=1 / scale,
        keep_ratio=True,
        keys=['gt', 'ref'],
        output_keys=['img', 'ref_down'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=float(scale),
        keep_ratio=True,
        keys=['img', 'ref_down'],
        output_keys=['img_lq', 'ref_lq'],
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackEditInputs')
]
demo_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(
        type='LoadImageFromFile',
        key='ref',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='ModCrop', key='img'),
    dict(
        type='Resize',
        scale=1 / scale,
        keep_ratio=True,
        keys=['ref'],
        output_keys=['ref_down'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=float(scale),
        keep_ratio=True,
        keys=['img', 'ref_down'],
        output_keys=['img_lq', 'ref_lq'],
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=9,
    batch_size=9,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='cufed', task_name='refsr'),
        data_root=data_root + '/CUFED',
        data_prefix=dict(ref='ref', gt='input'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='cufed', task_name='refsr'),
        data_root=data_root + '/CUFED',
        data_prefix=dict(ref='CUFED5', gt='CUFED5'),
        filename_tmpl=dict(ref='{}_1', gt='{}_0'),
        pipeline=valid_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR', crop_border=scale),
    dict(type='SSIM', crop_border=scale),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=200_000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))),
    extractor=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-5, betas=(0.9, 0.999))))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=False, milestones=[100000], gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
