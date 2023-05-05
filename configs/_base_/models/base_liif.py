_base_ = '../default_runtime.py'
work_dir = './work_dirs/liif'
save_dir = './work_dirs'

scale_min, scale_max = 1, 4
scale_test = 4

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='RandomDownSampling',
        scale_min=scale_min,
        scale_max=scale_max,
        patch_size=48),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='GenerateCoordinateAndCell', sample_quantity=2304),
    dict(type='PackInputs')
]
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='RandomDownSampling', scale_min=scale_max, scale_max=scale_max),
    dict(type='GenerateCoordinateAndCell', reshape_gt=False),
    dict(type='PackInputs')
]
# test_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         key='gt',
#         color_type='color',
#         channel_order='rgb',
#         imdecode_backend='cv2'),
#     dict(
#         type='LoadImageFromFile',
#         key='img',
#         color_type='color',
#         channel_order='rgb',
#         imdecode_backend='cv2'),
#     dict(type='GenerateCoordinateAndCell', scale=scale_test,
#          reshape_gt=False),
#     dict(type='PackInputs')
# ]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=8,
    batch_size=16,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DIV2K800sub_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',
        data_prefix=dict(gt='DIV2K_train_HR_sub'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',
        data_prefix=dict(img='LRbicx4', gt='GTmod12'),
        pipeline=val_pipeline))

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale_max),
        dict(type='SSIM', crop_border=scale_max),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1_000_000, val_interval=3000)
val_cfg = dict(type='MultiValLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    milestones=[200_000, 400_000, 600_000, 800_000],
    gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
