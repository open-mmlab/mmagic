_base_ = '../../default_runtime.py'

experiment_name = 'rdn_x2c64b16_g1_1000k_div2k'
work_dir = f'./work_dirs/{experiment_name}'

scale = 2
# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='RDNNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR'], crop_border=scale),
    data_preprocessor=dict(
        type='EditDataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=64),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data/DIV2K'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DIV2K800sub_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root,
        data_prefix=dict(
            img='DIV2K_train_LR_bicubic/X2_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}_x2', gt='{}'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root='data/Set5',
        data_prefix=dict(img='LRbicx2', gt='GTmod12'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR', crop_border=scale),
    dict(type='SSIM', crop_border=scale),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1000000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    milestones=[200000, 400000, 600000, 800000],
    gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=work_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
