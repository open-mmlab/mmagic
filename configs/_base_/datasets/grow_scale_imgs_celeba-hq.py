dataset_type = 'GrowScaleImgDataset'

pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackEditInputs')
]

train_dataloader = dict(
    num_workers=4,
    batch_size=64,
    dataset=dict(
        type=dataset_type,
        data_roots={
            '64': './data/celebahq/imgs_64',
            '256': './data/celebahq/imgs_256',
            '512': './data/celebahq/imgs_512',
            '1024': './data/celebahq/imgs_1024'
        },
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=300000,
        pipeline=pipeline),
    sampler=dict(type='InfiniteSampler', shuffle=True))

test_dataloader = dict(
    num_workers=4,
    batch_size=64,
    dataset=dict(
        type='BasicImageDataset',
        pipeline=pipeline,
        data_root='./data/celebahq/imgs_1024'),
    sampler=dict(type='DefaultSampler', shuffle=False))

val_dataloader = test_dataloader
