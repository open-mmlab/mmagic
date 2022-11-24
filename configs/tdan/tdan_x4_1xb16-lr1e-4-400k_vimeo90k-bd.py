_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/tdan_test_config.py'
]

experiment_name = 'tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bd'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'
scale = 4

# model settings
model = dict(
    type='TDAN',
    generator=dict(
        type='TDANNet',
        in_channels=3,
        mid_channels=64,
        out_channels=3,
        num_blocks_before_align=5,
        num_blocks_after_align=10),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    lq_pixel_loss=dict(type='MSELoss', loss_weight=0.01, reduction='mean'),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        std=[255, 255, 255],
        input_view=(1, -1, 1, 1),
        output_view=(-1, 1, 1)))

val_evaluator = [
    dict(type='PSNR', crop_border=8, convert_to='Y'),
    dict(type='SSIM', crop_border=8, convert_to='Y'),
]

train_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=192),
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
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackEditInputs')
]

data_root = 'data'

train_dataloader = dict(
    num_workers=8,
    batch_size=16,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vimeo_seq', task_name='vsr'),
        data_root=f'{data_root}/vimeo90k',
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_train_GT.txt',
        depth=2,
        num_input_frames=5,
        fixed_seq_len=7,
        load_frames_list=dict(
            img=['im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png'],
            gt=['im4.png']),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=f'{data_root}/Vid4',
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=2,
        num_input_frames=5,
        pipeline=val_pipeline))

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-6),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=400_000, val_interval=50000)
val_cfg = dict(type='ValLoop')

# No learning policy

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=50000,
        save_optimizer=True,
        out_dir=save_dir,
        by_epoch=False))
