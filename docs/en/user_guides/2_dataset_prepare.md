# Tutorial 2: Prepare Datasets

In this section, we will detail how to prepare data and adopt proper dataset in our repo for different methods.

The structure of this guide are as follows:

- [Download datasets](#download-datasets)
- [Prepare datasets](#prepare-datasets)
  - [Datasets for unconditional models](#datasets-for-unconditional-models)
  - [Datasets for image translation models](#datasets-for-image-translation-models)
- [Other datasets used in MMEditing](#other-datasets-used-in-mmediting)

## Download datasets

You are supposed to download datasets from their homepage first.
Most of datasets are available after downloaded, so you only need to make sure the folder structure is correct and further preparation is not necessary.
For example, you can simply prepare [Vimeo90K-triplet](./video_interpolation_datasets.md#Vimeo90K-triplet-Dataset) datasets by downloading datasets from [homepage](http://toflow.csail.mit.edu/).

## Prepare datasets

Some datasets need to be preprocessed before training or testing. We support many scripts to prepare datasets in [tools/data](/tools/data). And you can follow the tutorials of every dataset to run scripts.
For example, we recommend to crop the DIV2K images to sub-images. We provide a script to prepare cropped DIV2K dataset. You can run following command:

```shell
python tools/data/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

## Datasets for unconditional models

**Data preparation for unconditional model** is simple. What you need to do is downloading the images and put them into a directory. Next, you should set a symlink in the `data` directory. For standard unconditional gans with static architectures, like DCGAN and StyleGAN2, [UnconditionalImageDataset](<>) is designed to train such unconditional models. Here is an example config for FFHQ dataset:

```python
dataset_type = 'UnconditionalImageDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackGenInputs', keys=['img'], meta_keys=['img_path'])
]

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline))
```

Here, we adopt `InfinitySampler` to avoid frequent dataloader reloading, which will accelerate the training procedure. As shown in the example, `pipeline` provides important data pipeline to process images, including loading from file system, resizing, cropping, transferring to `torch.Tensor` and packing to `GenDataSample`. All of supported data pipelines can be found in `mmedit/datasets/transforms`.

For unconditional GANs with dynamic architectures like PGGAN and StyleGANv1, `GrowScaleImgDataset` is recommended to use for training. Since such dynamic architectures need real images in different scales, directly adopting `UnconditionalImageDataset` will bring heavy I/O cost for loading multiple high-resolution images. Here is an example we use for training PGGAN in CelebA-HQ dataset:

```python
dataset_type = 'GrowScaleImgDataset'

pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackGenInputs')
]

# `samples_per_gpu` and `imgs_root` need to be set.
train_dataloader = dict(
    num_workers=4,
    batch_size=64,
    dataset=dict(
        type='GrowScaleImgDataset',
        data_roots={
            '1024': './data/ffhq/images',
            '256': './data/ffhq/ffhq_imgs/ffhq_256',
            '64': './data/ffhq/ffhq_imgs/ffhq_64'
        },
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4,
            '128': 4,
            '256': 4,
            '512': 4,
            '1024': 4
        },
        len_per_stage=300000,
        pipeline=pipeline),
    sampler=dict(type='InfiniteSampler', shuffle=True))
```

In this dataset, you should provide a dictionary of image paths to the `data_roots`. Thus, you should resize the images in the dataset in advance.
For the resizing methods in the data pre-processing, we adopt bilinear interpolation methods in all of the experiments studied in MMEditing.

Note that this dataset should be used with `PGGANFetchDataHook`. In this config file, this hook should be added in the customized hooks, as shown below.

```python
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        # vis ema and orig at the same time
        vis_kwargs_list=dict(
            type='Noise',
            name='fake_img',
            sample_model='ema/orig',
            target_keys=['ema', 'orig'])),
    dict(type='PGGANFetchDataHook')
]
```

This fetching data hook helps the dataloader update the status of dataset to change the data source and batch size during training.

Here, we provide several download links of datasets frequently used in unconditional models: [LSUN](http://dl.yf.io/lsun/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [CelebA-HQ](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P), [FFHQ](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP).

### Datasets for image translation models

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
it into two images with the same shape in different domains. As shown in the example, `pipeline` provides important data pipeline to process images, including loading from file system, resizing, cropping, flipping, transferring to `torch.Tensor` and packing to `GenDataSample`. All of supported data pipelines can be found in `mmedit/datasets/transforms`.

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
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', io_backend='disk', key='img', flag='color'),
    dict(type='Resize', scale=(256, 256), interpolation='bicubic'),
    dict(
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
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

## Other datasets used in MMEditing

We support detailed tutorials and split them according to different tasks. Please follow the corresponding tutorials for data preparation of different tasks.

- [Prepare Inpainting Datasets](inpainting_datasets.md)
  - [Paris Street View](inpainting_datasets.md#paris-street-view-dataset) \[ [Homepage](https://github.com/pathak22/context-encoder/issues/24) \]
  - [CelebA-HQ](inpainting_datasets.md#celeba-hq-dataset) \[ [Homepage](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) \]
  - [Places365](inpainting_datasets.md#places365-dataset) \[ [Homepage](http://places2.csail.mit.edu/) \]
- [Prepare Matting Datasets](matting_datasets.md)
  - [Composition-1k](matting_datasets.md#composition-1k-dataset) \[ [Homepage](https://sites.google.com/view/deepimagematting) \]
- [Prepare Super-Resolution Datasets](super_resolution_datasets.md)
  - [DF2K_OST](super_resolution_datasets.md#df2kost-dataset) \[ [Homepage](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md) \]
  - [DIV2K](super_resolution_datasets.md#div2k-dataset) \[ [Homepage](https://data.vision.ee.ethz.ch/cvl/DIV2K/) \]
  - [REDS](super_resolution_datasets.md#reds-dataset) \[ [Homepage](https://seungjunnah.github.io/Datasets/reds.html) \]
  - [Vimeo90K](super_resolution_datasets.md#vimeo90k-dataset) \[ [Homepage](http://toflow.csail.mit.edu) \]
- [Prepare Video Frame Interpolation Datasets](video_interpolation_datasets.md)
  - [Vimeo90K-triplet](video_interpolation_datasets.md#vimeo90k-triplet-dataset) \[ [Homepage](http://toflow.csail.mit.edu) \]
