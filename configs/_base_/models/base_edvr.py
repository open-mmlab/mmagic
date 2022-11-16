_base_ = '../default_runtime.py'

scale = 4

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='TemporalReverse', keys='img_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
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
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(type='PackEditInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(type='PackEditInputs')
]

data_root = 'data/REDS'
save_dir = './work_dirs'

train_dataloader = dict(
    num_workers=8,
    batch_size=8,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_train.txt',
        depth=2,
        num_input_frames=5,
        num_output_frames=1,
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=2,
        num_input_frames=5,
        num_output_frames=1,
        pipeline=val_pipeline))

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
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)),
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        out_dir=save_dir,
        by_epoch=False))
