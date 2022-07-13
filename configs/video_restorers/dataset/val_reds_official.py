val_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(type='LoadImageFromFile', key='img', color_type='unchanged'),
    dict(type='LoadImageFromFileList', key='gt', color_type='unchanged'),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

val_dataloader = dict(
    num_workers=8,
    batch_size=8,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_official', task_name='vsr'),
        data_root='data/REDS',
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_official_val.txt',
        depth=2,
        num_input_frames=5,
        pipeline=val_pipeline))
