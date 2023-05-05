_base_ = './basicvsr_2xb4_reds4.py'

scale = 4
experiment_name = 'basicvsr_2xb4_vimeo90k-bd'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

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
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]

data_root = 'data'
file_list = [
    'im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png', 'im7.png'
]

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
        num_input_frames=7,
        fixed_seq_len=7,
        load_frames_list=dict(img=file_list, gt=file_list),
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_=True,
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
        depth=1,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='PSNR', convert_to='Y'),
        dict(type='SSIM', convert_to='Y'),
    ])

find_unused_parameters = True
