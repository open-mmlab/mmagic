# Design Your Own Data Pipelines

In this tutorial, we introduce the design of transforms pipeline in MMEditing.

The structure of this guide are as follows:

- [Data pipelines in MMEditing](#data-pipelines-in-mmediting)
  - [A simple example of data transform](#a-simple-example-of-data-transform)
  - [An example of BasicVSR](#an-example-of-basicvsr)
  - [An example of Pix2Pix](#an-example-of-pix2pix)
- [Supported transforms in MMEditing](#supported-transforms-in-mmediting)
  - [Data loading](#data-loading)
  - [Pre-processing](#pre-processing)
  - [Formatting](#formatting)
- [Extend and use custom pipelines](#extend-and-use-custom-pipelines)
  - [A simple example of MyTransform](#a-simple-example-of-mytransform)
  - [An example of flipping](#an-example-of-flipping)

## Data pipelines in MMEditing

Following typical conventions, we use `Dataset` and `DataLoader` for data loading with multiple workers. `Dataset` returns a dict of data items corresponding the arguments of models' forward method.

The data preparation pipeline and the dataset is decomposed. Usually a dataset defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.

A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

The operations are categorized into data loading, pre-processing, and formatting

In 1.x version of MMEditing, all data transformations are inherited from `BaseTransform`.
The input and output types of transformations are both dict.

### A simple example of data transform

```python
>>> from mmgen.transforms import LoadPairedImageFromFile
>>> transforms = LoadPairedImageFromFile(
>>>     key='pair',
>>>     domain_a='horse',
>>>     domain_b='zebra',
>>>     flag='color'),
>>> data_dict = {'pair_path': './data/pix2pix/facades/train/1.png'}
>>> data_dict = transforms(data_dict)
>>> print(data_dict.keys())
dict_keys(['pair_path', 'pair', 'pair_ori_shape', 'img_mask', 'img_photo', 'img_mask_path', 'img_photo_path', 'img_mask_ori_shape', 'img_photo_ori_shape'])
```

Generally, the last step of the transforms pipeline must be `PackGenInputs`.
`PackGenInputs` will pack the processed data into a dict containing two fields: `inputs` and `data_samples`.
`inputs` is the variable you want to use as the model's input, which can be the type of `torch.Tensor`, dict of `torch.Tensor`, or any type you want.
`data_samples` is a list of `GenDataSample`. Each `GenDataSample` contains groundtruth and necessary information for corresponding input.

### An example of BasicVSR

Here is a pipeline example for BasicVSR.

```python
train_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='MirrorSequence', keys=['img']),
    dict(type='PackEditInputs')
]
```

For each operation, we list the related dict fields that are added/updated/removed, the dict fields marked by '\*' are optional.

### An example of Pix2Pix

Here is a pipeline example for Pix2Pix training on aerial2maps dataset.

```python
source_domain = 'aerial'
target_domain = 'map'

pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': [f'img_{domain_a}', f'img_{domain_b}']},
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(
                type='mmgen.Resize', scale=(286, 286),
                interpolation='bicubic'),
            dict(type='mmgen.FixedCrop', crop_size=(256, 256))
        ]),
    dict(
        type='Flip',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        direction='horizontal'),
    dict(
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}', 'pair'],
        meta_keys=[
            'pair_path', 'sample_idx', 'pair_ori_shape',
            f'img_{domain_a}_path', f'img_{domain_b}_path',
            f'img_{domain_a}_ori_shape', f'img_{domain_b}_ori_shape', 'flip',
            'flip_direction'
        ])
]
```

## Supported transforms in MMEditing

### Data loading

`LoadImageFromFile`

- add: img, img_path, img_ori_shape, \*ori_img

`RandomLoadResizeBg`

- add: bg

`LoadMask`

- add: mask

`GetSpatialDiscountMask`

- add: discount_mask

### Pre-processing

`Resize`

- add: scale_factor, keep_ratio, interpolation, backend
- update: specified by `keys`

`MATLABLikeResize`

- add: scale, output_shape
- update: specified by `keys`

`RandomRotation`

- add: degrees
- update: specified by `keys`

`Flip`

- add: flip, flip_direction
- update: specified by `keys`

`RandomAffine`

- update: specified by `keys`

`RandomJitter`

- update: fg (img)

`ColorJitter`

- update: specified by `keys`

`BinarizeImage`

- update: specified by `keys`

`RandomMaskDilation`

- add: img_dilate_kernel_size

`RandomTransposeHW`

- add: transpose

`RandomDownSampling`

- update: scale, gt (img), lq (img)

`RandomBlur`

- update: specified by `keys`

`RandomResize`

- update: specified by `keys`

`RandomNoise`

- update: specified by `keys`

`RandomJPEGCompression`

- update: specified by `keys`

`RandomVideoCompression`

- update: specified by `keys`

`DegradationsWithShuffle`

- update: specified by `keys`

`GenerateFrameIndices`

- update: img_path (gt_path, lq_path)

`GenerateFrameIndiceswithPadding`

- update: img_path (gt_path, lq_path)

`TemporalReverse`

- add: reverse
- update: specified by `keys`

`GenerateSegmentIndices`

- add: interval
- update: img_path (gt_path, lq_path)

`MirrorSequence`

- update: specified by `keys`

`CopyValues`

- add: specified by `dst_key`

`UnsharpMasking`

- add: img_unsharp

`Crop`

- add: img_crop_bbox, crop_size
- update: specified by `keys`

`RandomResizedCrop`

- add: img_crop_bbox
- update: specified by `keys`

`FixedCrop`

- add: crop_size, crop_pos
- update: specified by `keys`

`PairedRandomCrop`

- update: gt (img), lq (img)

`CropAroundCenter`

- add: crop_bbox
- update: fg (img), alpha (img), trimap (img), bg (img)

`CropAroundUnknown`

- add: crop_bbox
- update: specified by `keys`

`CropAroundFg`

- add: crop_bbox
- update: specified by `keys`

`ModCrop`

- update: gt (img)

`CropLike`

- update: specified by `target_key`

`GetMaskedImage`

- add: masked_img

`GenerateFacialHeatmap`

- add: heatmap

`GenerateCoordinateAndCell`

- add: coord, cell
- update: gt (img)

`Normalize`

- add: img_norm_cfg
- update: specified by `keys`

`RescaleToZeroOne`

- update: specified by `keys`

...

### Formatting

`ToTensor`

- update: specified by `keys`.

`FormatTrimap`

- update: trimap

`PackEditInputs`

- add: inputs, data_sample
- remove: all other keys

## Extend and use custom pipelines

### A simple example of MyTransform

1. Write a new pipeline in a file, e.g., in `my_pipeline.py`. It takes a dict as input and returns a dict.

```python
import random
from mmcv.transforms import BaseTransform
from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MyTransform(BaseTransform):
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def transform(self, results):
        if random.random() > self.p:
            results['dummy'] = True
        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(p={self.p})')

        return repr_str
```

2. Import and use the pipeline in your config file.

Make sure the import is relative to where your train script is located.

```python
train_pipeline = [
    ...
    dict(type='MyTransform', p=0.2),
    ...
]
```

### An example of flipping

Here we use a simple flipping transformation as example:

```python
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

Thus, we can instantiate a `MyFlip` object and use it to process the data dict.

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

Or, we can use `MyFlip` transformation in data pipeline in our config file.

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

Note that if you want to use `MyFlip` in config, you must ensure the file containing `MyFlip` is imported during the program run.
