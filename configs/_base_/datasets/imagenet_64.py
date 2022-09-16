# dataset settings
dataset_type = 'ImageNet'

# different from mmcls, we adopt the setting used in BigGAN.
# We use `RandomCropLongEdge` in training and `CenterCropLongEdge` in testing.
train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='RandomCropLongEdge', keys=['img']),
    dict(type='Resize', scale=(64, 64), keys=['img'], backend='pillow'),
    dict(type='Flip', flip_ratio=0.5, direction='horizontal'),
    dict(type='PackEditInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', scale=(64, 64), backend='pillow'),
    dict(type='PackEditInputs')
]

train_dataloader = dict(
    batch_size=None,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='./data/imagenet/',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='./data/imagenet/',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = val_dataloader
