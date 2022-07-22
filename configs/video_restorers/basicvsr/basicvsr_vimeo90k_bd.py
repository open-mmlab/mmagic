_base_ = './basicvsr_reds4.py'

scale = 4
experiment_name = 'basicvsr_vimeo90k_bd'
work_dir = f'./work_dirs/{experiment_name}'

train_pipeline = [
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
    dict(type='MirrorSequence', keys=['img', 'gt']),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='MirrorSequence', keys=['img']),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackEditInputs')
]

data_root = 'openmmlab:s3://openmmlab/datasets/editing'

train_dataloader = dict(
    num_workers=6,
    batch_size=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=f'{data_root}/vimeo90k',
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_train_GT.txt',
        depth=2,
        fixed_seq_len=7,
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
        num_input_frames=100,
        fixed_seq_len=100,
        pipeline=val_pipeline))

test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=f'{data_root}/vimeo90k',
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_test_GT.txt',
        depth=2,
        num_input_frames=7,
        fixed_seq_len=7,
        pipeline=test_pipeline))

val_evaluator = [
    dict(type='PSNR', convert_to='Y'),
    dict(type='SSIM', convert_to='Y'),
]
test_evaluator = val_evaluator
