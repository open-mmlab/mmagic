from mmagic.datasets.imagenet_dataset import ImageNet
from mmagic.datasets.transforms.loading import LoadImageFromFile
# from mmcv.transforms.loading import LoadImageFromFile
# from mmcv.transforms.loading import LoadImageFromFile
from mmagic.datasets.transforms.crop import RandomCropLongEdge
from mmagic.datasets.transforms.crop import CenterCropLongEdge
from mmagic.datasets.transforms.aug_shape import Resize 
from mmagic.datasets.transforms.aug_shape import Flip
from mmagic.datasets.transforms.formatting import PackInputs
from mmengine.dataset.sampler import DefaultSampler
# dataset settings
dataset_type = ImageNet

# different from mmcls, we adopt the setting used in BigGAN.
# We use `RandomCropLongEdge` in training and `CenterCropLongEdge` in testing.
train_pipeline = [
    dict(type=LoadImageFromFile, key='gt'),
    dict(type=RandomCropLongEdge, keys='gt'),
    dict(type=Resize, scale=(64, 64), keys='gt', backend='pillow'),
    dict(type=Flip, keys='gt', flip_ratio=0.5, direction='horizontal'),
    dict(type=PackInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, key='gt'),
    dict(type=CenterCropLongEdge, keys='gt'),
    dict(type=Resize, scale=(64, 64), keys='gt', backend='pillow'),
    dict(type=PackInputs)
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
    sampler=dict(type=DefaultSampler, shuffle=True),
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
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

test_dataloader = val_dataloader
