# Image Translation Datasets

**Data preparation for translation model** needs a little attention. You should organize the files in the way we told you in `quick_run.md`. Fortunately, for most official datasets like facades and summer2winter_yosemite, they already have the right format. Also, you should set a symlink in the `data` directory. For paired-data trained translation model like Pix2Pix , `PairedImageDataset` is designed to train such translation models. Here is an example config for facades dataset:

```python
train_dataset_type = 'PairedImageDataset'
val_dataset_type = 'PairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(286, 286),
        interpolation='bicubic')
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='image',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(256, 256),
        interpolation='bicubic')
]
dataroot = 'data/paired/facades'
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=dataroot,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataroot,  # set by user
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataroot,  # set by user
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
```

Here, we adopt `LoadPairedImageFromFile` to load a paired image as the common loader does and crops
it into two images with the same shape in different domains. As shown in the example, `pipeline` provides important data pipeline to process images, including loading from file system, resizing, cropping, flipping, transferring to `torch.Tensor` and packing to `EditDataSample`. All of supported data pipelines can be found in `mmedit/datasets/transforms`.

For unpaired-data trained translation model like CycleGAN , `UnpairedImageDataset` is designed to train such translation models. Here is an example config for horse2zebra dataset:

```python
train_dataset_type = 'UnpairedImageDataset'
val_dataset_type = 'UnpairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
domain_a, domain_b = 'horse', 'zebra'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': [f'img_{domain_a}', f'img_{domain_b}']},
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(type='Resize', scale=(286, 286), interpolation='bicubic'),
            dict(type='Crop', crop_size=(256, 256), random_crop=True),
        ]),
    dict(type='Flip', keys=[f'img_{domain_a}'], direction='horizontal'),
    dict(type='Flip', keys=[f'img_{domain_b}'], direction='horizontal'),
    dict(
        type='PackEditInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', io_backend='disk', key='img', flag='color'),
    dict(type='Resize', scale=(256, 256), interpolation='bicubic'),
    dict(
        type='PackEditInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}'])
]
data_root = './data/horse2zebra/'
# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=None,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,  # set by user
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=None,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,  # set by user
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
```

`UnpairedImageDataset` will load both images (domain A and B) from different paths and transform them at the same time.

Here, we provide download links of datasets used in [Pix2Pix](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) and [CycleGAN](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).
