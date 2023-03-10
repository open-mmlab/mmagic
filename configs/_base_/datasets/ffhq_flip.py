dataset_type = 'BasicImageDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='Flip', keys=['gt'], direction='horizontal'),
    dict(type='PackEditInputs', keys='gt')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='PackEditInputs', keys=['gt'])
]

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(gt=''),
        data_root=None,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(gt=''),
        data_root=None,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(gt=''),
        data_root=None,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
