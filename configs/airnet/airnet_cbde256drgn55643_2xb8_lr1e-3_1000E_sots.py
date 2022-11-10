_base_ = '../_base_/default_runtime.py'

experiment_name = 'airnet'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
# number of degradations
# should be set equal to the number of train datasets
num_degra = 8
# in paper, it is 400 * N degradations in each epoch
# which means 400 batches each epoch
epoch_image_size = 400 * num_degra

model = dict(
    type='AirNetRestorer',
    generator=dict(
        type='AirNet',
        encoder_cfg=dict(
            type='CBDE',
            batch_size=num_degra,
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
    train_cfg=dict(epochs_encoder=100),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ),
    train_patch_size=128)

find_unused_parameters = True

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

train_dataloader = dict(
    num_workers=8,
    batch_size=num_degra,  # gpus 4
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='PairedMultipathDataset',
        fix_length=400,
        datasets=[
            dict(
                type=dataset_type,
                data_root='../datasets/SIDD/train',
                data_prefix=dict(gt='gt', img='noisy'),
                filename_tmpl=dict(img='{}_NOISY', gt='{}_GT'),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root='../datasets/RainTrainL',
                data_prefix=dict(gt='gt', img='input'),
                pipeline=train_pipeline),
        ]))

val_dataloader = dict(
    num_workers=4,
    batch_size=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='SOTS', task_name='dehaze'),
        data_root='../datasets/SOTS/outdoor',
        data_prefix=dict(gt='gt', img='input'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000)
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
        interval=10,
        save_optimizer=True,
        by_epoch=True,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
