test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='spmcs', task_name='vsr'),
        data_root='data/SPMCS',
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_SPMCS_GT.txt',
        depth=2,
        num_input_frames=5,
        pipeline=test_pipeline))
