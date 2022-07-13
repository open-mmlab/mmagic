_base_ = '../../default_runtime.py'

experiment_name = 'basicvsr_reds4'
work_dir = f'./work_dirs/{experiment_name}'

scale = 4

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRNet',
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRREDSMultipleGTDataset'
val_dataset_type = 'SRREDSMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
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
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackEditInputs')
]

train_dataloader = dict(
    num_workers=6,
    batch_size=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root='data/REDS',
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_train.txt',
        depth=2,
        num_input_frames=15,
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root='data/REDS',
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=2,
        num_input_frames=100,
        pipeline=train_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300_000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    optimizer=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99))),
    paramwise_cfg=dict(custom_keys={'.extractor': dict(lr_mult=0.125)}))

# # learning policy
# total_iters = 300000
# lr_config = dict(
#     policy='CosineRestart',
#     by_epoch=False,
#     periods=[300000],
#     restart_weights=[1],
#     min_lr=1e-7)
