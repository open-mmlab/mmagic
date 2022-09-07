dataset_type = 'GrowScaleImgDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Resize', scale=(128, 128)),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackEditInputs')
]

train_dataloader = dict(
    num_workers=4,
    batch_size=64,  # initialize batch size
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_roots={},
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=-1),
    sampler=dict(type='InfiniteSampler', shuffle=True))

test_dataloader = dict(
    num_workers=4,
    batch_size=64,
    dataset=dict(
        type='UnconditionalImageDataset',
        pipeline=train_pipeline,
        data_root=None),
    sampler=dict(type='DefaultSampler', shuffle=False))

val_dataloader = test_dataloader
