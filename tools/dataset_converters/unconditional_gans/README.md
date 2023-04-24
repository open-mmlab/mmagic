# Unconditional GANs Datasets

**Data preparation for unconditional model** is simple. What you need to do is downloading the images and put them into a directory. Next, you should set a symlink in the `data` directory. For standard unconditional gans with static architectures, like DCGAN and StyleGAN2, `UnconditionalImageDataset` is designed to train such unconditional models. Here is an example config for FFHQ dataset:

```python
dataset_type = 'BasicImageDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackInputs', keys=['img'], meta_keys=['img_path'])
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

Here, we adopt `InfinitySampler` to avoid frequent dataloader reloading, which will accelerate the training procedure. As shown in the example, `pipeline` provides important data pipeline to process images, including loading from file system, resizing, cropping, transferring to `torch.Tensor` and packing to `DataSample`. All of supported data pipelines can be found in `mmagic/datasets/transforms`.

For unconditional GANs with dynamic architectures like PGGAN and StyleGANv1, `GrowScaleImgDataset` is recommended to use for training. Since such dynamic architectures need real images in different scales, directly adopting `UnconditionalImageDataset` will bring heavy I/O cost for loading multiple high-resolution images. Here is an example we use for training PGGAN in CelebA-HQ dataset:

```python
dataset_type = 'GrowScaleImgDataset'

pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackInputs')
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
For the resizing methods in the data pre-processing, we adopt bilinear interpolation methods in all of the experiments studied in MMagic.

Note that this dataset should be used with `PGGANFetchDataHook`. In this config file, this hook should be added in the customized hooks, as shown below.

```python
custom_hooks = [
    dict(
        type='VisualizationHook',
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
