_base_ = [
    '../_base_/default_runtime.py'
]

save_dir = './work_dir/'

model = dict(
    type='DeblurGanV2',
    generator=dict(
        type='FPNInception',
        norm_layer='instance',
        output_ch=3,
        num_filters=128,
        num_filters_fpn=256,
    ),
    discriminator=dict(
        type='DoubleGan',
        norm_layer='instance',
        d_layers=3,
    ),
    pixel_loss='perceptual',
    disc_loss='wgan-gp',
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.5, 0.5, 0.5],
        std=[255, 255, 255],
    )
)

# DistributedDataParallel
#model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='SetValues', dictionary=dict(scale=1)),
    # dict(
    #     type='Flip',
    #     keys=['img', 'gt'],
    #     flip_ratio=0.5,
    #     direction='horizontal'),
    #dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    # dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Crop',
        keys=['img', 'gt'],
        crop_size=(256, 256),
        random_crop=False),
    # dict(type='PairedAlbuTransForms',
    #      size=256,
    #      keys=['img', 'gt']),
    # dict(type='AlbuCorruptFunction',
    #      keys=['img', 'gt'],
    #      config=[
    #          {
    #              'name': 'cutout',
    #              'prob': 0.5,
    #              'num_holes': 3,
    #              'max_h_size': 25,
    #              'max_w_size': 25
    #          },
    #          {'name': 'jpeg', 'quality_lower': 70, 'quality_upper': 90},
    #          {'name': 'motion_blur'},
    #          {'name': 'median_blur'},
    #          {'name': 'gamma'},
    #          {'name': 'rgb_shift'},
    #          {'name': 'hsv_shift'},
    #          {'name': 'sharpen'}]),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

test_pipeline = val_pipeline

data_root = 'G:/github/DeblurGANv2/data/gopro'
train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='G:/github/DeblurGANv2/data/gopro/debug',
        data_prefix=dict(img='input', gt='target'),
        #ann_file='meta_info_gopro_train.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='G:/github/DeblurGANv2/data/gopro/debug',
        # ann_file='meta_info_gopro_test.txt',
        data_prefix=dict(img='input', gt='target'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=10000, val_interval=100)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiValLoop')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))),
)

# learning policy
param_scheduler = dict(
    type='LinearLrInterval',
    interval=100,
    by_epoch=False,
    start_factor=0.0002,
    end_factor=0,
    begin=50,
    end=10000)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

