# Base config for CelebA-HQ dataset

# dataset settings
dataset_type = 'BasicImageDataset'
# data_root = 'openmmlab:s3://openmmlab/datasets/editing/CelebA-HQ'
# data_root = 's3://openmmlab/datasets/editing/CelebA-HQ'
data_root = 'data/CelebA-HQ'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(gt='data_large'),
        ann_file='train_celeba_img_list.txt',
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
        ann_file='val_celeba_img_list.txt',
        test_mode=True,
    ))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),  # L1Loss
    dict(type='PSNR'),
    dict(type='SSIM'),
]

test_evaluator = val_evaluator
