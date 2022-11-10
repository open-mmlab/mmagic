_base_ = '../_base_/default_runtime.py'

experiment_name = 'cain_g1b32_1xb5_vimeo90k-triplet'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# model settings
model = dict(
    type='CAIN',
    generator=dict(type='CAINNet'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    required_frames=2,
    step_frames=1,
    init_cfg=None,
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        input_view=(1, -1, 1, 1),
        output_view=(-1, 1, 1),
        pad_args=dict(mode='reflect')))

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
    dict(type='FixedCrop', keys=['img', 'gt'], crop_size=(256, 256)),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(
        type='ColorJitter',
        keys=['img', 'gt'],
        channel_order='rgb',
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.05),
    dict(type='TemporalReverse', keys=['img'], reverse_ratio=0.5),
    dict(type='PackEditInputs')
]

val_pipeline = [
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
    dict(type='PackEditInputs')
]

demo_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='PackEditInputs')
]

# dataset settings
train_dataset_type = 'BasicFramesDataset'
val_dataset_type = 'BasicFramesDataset'
data_root = 'data/vimeo_triplet'

train_dataloader = dict(
    num_workers=32,
    batch_size=32,  # 1 gpu
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
        pipeline=val_pipeline,
        depth=2,
        load_frames_list=dict(img=['im1.png', 'im3.png'], gt=['im2.png'])))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
)

# learning policy
param_scheduler = dict(
    type='ReduceLR',
    by_epoch=True,
    mode='min',
    factor=0.5,
    patience=5,
    cooldown=0)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_optimizer=True,
        by_epoch=True,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='EditVisualizationHook'),
    param_scheduler=dict(
        type='ReduceLRSchedulerHook',
        by_epoch=True,
        interval=1,
        val_metric='MAE'),
)

log_processor = dict(type='LogProcessor', by_epoch=True)
