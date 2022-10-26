dataset_type = 'BasicImageDataset'

# TODO:
train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='Flip', keys=['img'], direction='horizontal'),  # TODO:
    dict(type='PackEditInputs')
]

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=None,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=None,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=None,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
