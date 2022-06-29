_base_ = '../../default_runtime.py'

exp_name = 'flavr_in4out1_g8b4_vimeo90k_septuplet'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='BasicInterpolator',
    generator=dict(
        type='FLAVRNet',
        num_input_frames=4,
        num_output_frames=1,
        mid_channels_list=[512, 256, 128, 64],
        encoder_layers_list=[2, 2, 2, 2],
        bias=False,
        norm_cfg=None,
        join_type='concat',
        up_mode='transpose'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=None,
    test_cfg=None,
    required_frames=4,
    step_frames=1,
    init_cfg=None,
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        input_view=(1, -1, 1, 1),
        output_view=(-1, 1, 1),
    ))

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
    dict(type='ToTensor', keys=['img', 'gt']),
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
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

demo_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        channel_order='rgb',
        imdecode_backend='pillow'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackEditInputs')
]

# dataset settings
train_dataset_type = 'BasicFramesDataset'
val_dataset_type = 'BasicFramesDataset'
data_root = 'data/vimeo90k_septenary'

train_dataloader = dict(
    num_workers=16,
    batch_size=4,  # 8 gpu
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=train_dataset_type,
        ann_file='txt/sep_trainlist.txt',
        metainfo=dict(dataset_type='vimeo90k_septenary', task_name='vfi'),
        data_root=data_root,
        data_prefix=dict(img='GT', gt='GT'),
        pipeline=train_pipeline,
        depth=2,
        load_frames_list=dict(
            img=['im1.png', 'im3.png', 'im5.png', 'im7.png'], gt=['im4.png'])))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_dataset_type,
        ann_file='txt/sep_testlist.txt',
        metainfo=dict(dataset_type='vimeo90k_septenary', task_name='vfi'),
        data_root=data_root,
        data_prefix=dict(img='GT', gt='GT'),
        pipeline=val_pipeline,
        depth=2,
        load_frames_list=dict(
            img=['im1.png', 'im3.png', 'im5.png', 'im7.png'], gt=['im4.png'])))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

epoch_length = 2020

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1_000_000, val_interval=epoch_length)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    ))

# learning policy
# 1604 iters == 1 epoch
total_iters = 1000000
lr_config = dict(
    type='ReduceLR',
    by_epoch=False,
    mode='min',
    factor=0.5,
    patience=10,
    cooldown=20,
    verbose=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=epoch_length * 2,
        save_optimizer=True,
        by_epoch=False),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='EditVisualizationHook'),
    param_scheduler=dict(
        type='ReduceLRSchedulerHook',
        by_epoch=False,
        interval=epoch_length,
        val_metric='MAE'),
)
