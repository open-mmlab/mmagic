# Base config for places365 dataset

# dataset settings
dataset_type = 'BasicImageDataset'
# data_root = 'openmmlab:s3://openmmlab/datasets/editing/Places'
# data_root = 's3://openmmlab/datasets/editing/Places'
data_root = 'data/Places'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(gt='data_large'),
        ann_file='meta/places365_train_challenge.txt',
        # Note that Places365-standard (1.8M images) and
        # Place365-challenge (8M images) use different image lists.
        test_mode=False,
    ))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(gt='val_large'),
        ann_file='meta/places365_val.txt',
        test_mode=True,
    ))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),  # L1Loss
    dict(type='PSNR'),
    dict(type='SSIM'),
]

test_evaluator = val_evaluator
