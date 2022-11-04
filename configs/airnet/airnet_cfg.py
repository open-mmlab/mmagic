_base_ = '../_base_/default_runtime.py'

experiment_name = 'airnet'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='AirNetModel',
    generator=dict(
        type='AirNet',
        encoder_cfg=dict(
            type='CBDE',
            batch_size=8,
            dim=256,
        ),
        restorer_cfg=dict(
            type='DGRN',
            n_groups=5,
            n_blocks=5,
            n_feats=64,
            kernel_size=3,
        ),
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        input_view=[0., 0., 0.],
        output_view=[255., 255., 255.]),
    train_patch_size=128)

train_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='FixedCrop', keys=['img', 'gt'], crop_size=(256, 256)),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
N = 1

train_dataloader = dict(
    num_workers=8,
    batch_size=400 * N,  # gpus 4
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='../../datasets/gopro/train',
        data_prefix=dict(gt='sharp', img='blur'),
        ann_file='meta_info_gopro_train.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='../../datasets/gopro/test',
        ann_file='meta_info_gopro_test.txt',
        data_prefix=dict(gt='sharp', img='blur'),
        pipeline=val_pipeline))

test_dataloader = dict(
    num_workers=4,
    batch_size=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='../../datasets/gopro/mini_test',
        data_prefix=dict(gt='sharp', img='blur'),
        pipeline=val_pipeline))

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=1000, epochs_encoder=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3, betas=(0.9, 0.999)),
)

# learning policy
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[60], gamma=0.1),
    dict(type='StepLR', by_epoch=True, step_size=125, gamma=0.5, begin=100)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_optimizer=True,
        by_epoch=True,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
