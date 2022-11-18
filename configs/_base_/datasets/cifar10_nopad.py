cifar_pipeline = [
    dict(type='Flip', keys=['img'], flip_ratio=0.5, direction='horizontal'),
    dict(type='PackEditInputs')
]
cifar_dataset = dict(
    type='CIFAR10',
    data_root='./data',
    data_prefix='cifar10',
    test_mode=False,
    pipeline=cifar_pipeline)

train_dataloader = dict(
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
