_base_ = '../_base_/default_runtime.py'

experiment_name = 'ifrnet_in2out7_8xb4_gopro'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='IFRNet',
    generator=dict(type='IFRNetInterpolator'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    interpolation_scale=8,
    required_frames=2,
    step_frames=1,
    init_cfg=None,
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        # input_view=(1, -1, 1, 1),
        # output_view=(-1, 1, 1),
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
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='TemporalReverse', keys=['img'], reverse_ratio=0.5),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='FixedCrop', keys=['img', 'gt'], crop_size=(512, 512)),
    dict(type='PackEditInputs')
]

# dataset settings
train_dataset_type = 'MultipleFramesDataset'
val_dataset_type = 'MultipleFramesDataset'
data_root = '../datasets/Adobe240fps/'

train_dataloader = dict(
    num_workers=8,
    batch_size=4,  # 8 gpu
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=train_dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='mfi'),
        data_root=data_root + 'train',
        data_prefix=dict(img='full_sharp', gt='full_sharp'),
        pipeline=train_pipeline,
        depth=2,
        load_frames_list=dict(img=[0, 8], gt=[1, 2, 3, 4, 5, 6, 7])))

val_dataloader = dict(
    num_workers=8,
    batch_size=8,  # 8 gpu
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='mfi'),
        data_root=data_root + 'test',
        data_prefix=dict(img='full_sharp', gt='full_sharp'),
        pipeline=val_pipeline,
        depth=2,
        load_frames_list=dict(img=[0, 8], gt=[1, 2, 3, 4, 5, 6, 7])))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0),
)

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=False, T_max=600_000, eta_min=1e-5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,
        save_optimizer=True,
        by_epoch=True,
        out_dir=save_dir,
        max_keep_ckpts=10,
        save_best='PSNR',
        rule='greater',
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='EditVisualizationHook'),
)

visualizer = dict(img_keys=['input', 'gt_img', 'pred_img'], fn_key='key')
log_processor = dict(type='LogProcessor', by_epoch=True)
