_base_ = [
    '../../default_runtime.py',
    '../dataset/reds_reds4_train.py',
    '../dataset/reds_reds4_val.py',
]

scale = 4

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='TemporalReverse', keys='img_path', reverse_ratio=0),
    dict(type='LoadImageFromFile', key='img', color_type='unchanged'),
    dict(type='LoadImageFromFile', key='gt', color_type='unchanged'),
    dict(type='PairedRandomCrop', gt_patch_size=256),
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

val_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(type='LoadImageFromFile', key='img', color_type='unchanged'),
    dict(type='LoadImageFromFileList', key='gt', color_type='unchanged'),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', color_type='unchanged'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackEditInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=600_000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)),
    ))

# learning policy
# lr_config = dict(
#     policy='CosineRestart',
#     by_epoch=False,
#     periods=[50000, 100000, 150000, 150000, 150000],
#     restart_weights=[1, 1, 1, 1, 1],
#     min_lr=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        out_dir='s3://ysli/edvr/',
        by_epoch=False))
