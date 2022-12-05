_base_ = '../_base_/default_runtime.py'

experiment_name = 'swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SwinIRNet',
        upscale=1,
        in_chans=1,
        img_size=126,
        window_size=7,
        img_range=255.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='',
        resi_connection='1conv'),
    pixel_loss=dict(type='CharbonnierLoss', eps=1e-9),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(type='EditDataPreprocessor', mean=[0.], std=[255.]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='PairedRandomCrop', gt_patch_size=126),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 30], color_type='grayscale'),
        keys=['img']),
    dict(type='PackEditInputs')
]
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 30], color_type='grayscale'),
        keys=['img']),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=4,
    batch_size=1,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DFWB8550sub_GT.txt',
        metainfo=dict(dataset_type='dfwb', task_name='gray_CAR_10'),
        data_root=data_root + '/DFWB',
        data_prefix=dict(img='', gt=''),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='classic5', task_name='gray_CAR_10'),
        data_root=data_root + '/classic5',
        data_prefix=dict(img='', gt=''),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]

test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1_600_000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    milestones=[800000, 1200000, 1400000, 1500000, 1600000],
    gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
