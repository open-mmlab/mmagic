_base_ = '../default_runtime.py'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='PackInputs')
]

# dataset settings
train_dataset_type = 'BasicFramesDataset'
val_dataset_type = 'BasicFramesDataset'
data_root = 'data/vimeo_triplet'
save_dir = './work_dirs'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=train_dataset_type,
        ann_file='tri_trainlist.txt',
        metainfo=dict(dataset_type='vimeo90k', task_name='vfi'),
        data_root=data_root,
        data_prefix=dict(img='sequences', gt='sequences'),
        pipeline=train_pipeline,
        depth=2,
        load_frames_list=dict(img=['im1.png', 'im3.png'], gt=['im2.png'])))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_dataset_type,
        ann_file='tri_testlist.txt',
        metainfo=dict(dataset_type='vimeo90k', task_name='vfi'),
        data_root=data_root,
        data_prefix=dict(img='sequences', gt='sequences'),
        pipeline=train_pipeline,
        depth=2,
        load_frames_list=dict(img=['im1.png', 'im3.png'], gt=['im2.png'])))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

# 5000 iters == 1 epoch
epoch_length = 5000

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1_000_000, val_interval=epoch_length)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=5e-5,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    ),
)

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    gamma=0.5,
    milestones=[200000, 400000, 600000, 800000])

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=epoch_length,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
