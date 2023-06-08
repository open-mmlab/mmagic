# How to design your own data transforms

In this tutorial, we introduce the design of transforms pipeline in MMagic.

The structure of this guide are as follows:

- [How to design your own data transforms](#how-to-design-your-own-data-transforms)
  - [Data pipelines in MMagic](#data-pipelines-in-mmagic)
    - [A simple example of data transform](#a-simple-example-of-data-transform)
    - [An example of BasicVSR](#an-example-of-basicvsr)
    - [An example of Pix2Pix](#an-example-of-pix2pix)
  - [Supported transforms in MMagic](#supported-transforms-in-mmagic)
    - [Data loading](#data-loading)
    - [Pre-processing](#pre-processing)
    - [Formatting](#formatting)
  - [Extend and use custom pipelines](#extend-and-use-custom-pipelines)
    - [A simple example of MyTransform](#a-simple-example-of-mytransform)
    - [An example of flipping](#an-example-of-flipping)

## Data pipelines in MMagic

Following typical conventions, we use `Dataset` and `DataLoader` for data loading with multiple workers. `Dataset` returns a dict of data items corresponding the arguments of models' forward method.

The data preparation pipeline and the dataset is decomposed. Usually a dataset defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.

A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

The operations are categorized into data loading, pre-processing, and formatting

In MMagic, all data transformations are inherited from `BaseTransform`.
The input and output types of transformations are both dict.

### A simple example of data transform

```python
>>> from mmagic.transforms import LoadPairedImageFromFile
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

Generally, the last step of the transforms pipeline must be `PackInputs`.
`PackInputs` will pack the processed data into a dict containing two fields: `inputs` and `data_samples`.
`inputs` is the variable you want to use as the model's input, which can be the type of `torch.Tensor`, dict of `torch.Tensor`, or any type you want.
`data_samples` is a list of `DataSample`. Each `DataSample` contains groundtruth and necessary information for corresponding input.

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
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='MirrorSequence', keys=['img']),
    dict(type='PackInputs')
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
                type='mmagic.Resize', scale=(286, 286),
                interpolation='bicubic'),
            dict(type='mmagic.FixedCrop', crop_size=(256, 256))
        ]),
    dict(
        type='Flip',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        direction='horizontal'),
    dict(
        type='PackInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}', 'pair'])
```

## Supported transforms in MMagic

### Data loading

<table class="docutils">
   <thead>
      <tr>
         <th style="text-align:center">Transform</th>
         <th style="text-align:center">Modification of Results' keys</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>
            <code>LoadImageFromFile</code>
         </td>
         <td>
            - add: img, img_path, img_ori_shape, \*ori_img
         </td>
      </tr>
      <tr>
         <td>
            <code>RandomLoadResizeBg</code>
         </td>
         <td>
            - add: bg
         </td>
      </tr>
      <tr>
         <td>
            <code>LoadMask</code>
         </td>
         <td>
            - add: mask
         </td>
      </tr>
      <tr>
         <td>
            <code>GetSpatialDiscountMask</code>
         </td>
         <td>
            - add: discount_mask
         </td>
      </tr>
   </tbody>
</table>

### Pre-processing

<table class="docutils">
   <thead>
      <tr>
         <th style="text-align:center">Transform</th>
         <th style="text-align:center">Modification of Results' keys</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>
            <code>Resize</code>
         </td >
         <td>
            - add: scale_factor, keep_ratio, interpolation, backend
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>MATLABLikeResize</code>
         </td >
         <td>
            - add: scale, output_shape
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomRotation</code>
         </td >
         <td>
            - add: degrees
            - update: specified by <code>keys</code>
         <td>
      </tr>
      <tr>
         <td>
            <code>Flip</code>
         </td >
         <td>
            - add: flip, flip_direction
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomAffine</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomJitter</code>
         </td >
         <td>
            - update: fg (img)
         </td >
      </tr>
      <tr>
         <td>
            <code>ColorJitter</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>BinarizeImage</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomMaskDilation</code>
         </td >
         <td>
            - add: img_dilate_kernel_size
         <td>
      </tr>
      <tr>
         <td>
            <code>RandomTransposeHW</code>
         </td >
         <td>
            - add: transpose
      </tr>
      <tr>
         <td>
            <code>RandomDownSampling</code>
         </td >
         <td>
            - update: scale, gt (img), lq (img)
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomBlur</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomResize</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomNoise</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomJPEGCompression</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomVideoCompression</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>DegradationsWithShuffle</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>GenerateFrameIndices</code>
         </td >
         <td>
            - update: img_path (gt_path, lq_path)
         </td >
      </tr>
      <tr>
         <td>
            <code>GenerateFrameIndiceswithPadding</code>
         </td >
         <td>
            - update: img_path (gt_path, lq_path)
         </td >
      </tr>
      <tr>
         <td>
            <code>TemporalReverse</code>
         </td >
         <td>
            - add: reverse
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>GenerateSegmentIndices</code>
         </td >
         <td>
            - add: interval
            - update: img_path (gt_path, lq_path)
         </td >
      </tr>
      <tr>
         <td>
            <code>MirrorSequence</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>CopyValues</code>
         </td >
         <td>
            - add: specified by <code>dst_key</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>UnsharpMasking</code>
         </td >
         <td>
            - add: img_unsharp
         </td >
      </tr>
      <tr>
         <td>
            <code>Crop</code>
         </td >
         <td>
            - add: img_crop_bbox, crop_size
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RandomResizedCrop</code>
         </td >
         <td>
            - add: img_crop_bbox
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>FixedCrop</code>
         </td >
         <td>
            - add: crop_size, crop_pos
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>PairedRandomCrop</code>
         </td >
         <td>
            - update: gt (img), lq (img)
         </td >
      </tr>
      <tr>
         <td>
            <code>CropAroundCenter</code>
         </td >
         <td>
            - add: crop_bbox
            - update: fg (img), alpha (img), trimap (img), bg (img)
         </td >
      </tr>
      <tr>
         <td>
            <code>CropAroundUnknown</code>
         </td >
         <td>
            - add: crop_bbox
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>CropAroundFg</code>
         </td >
         <td>
            - add: crop_bbox
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>ModCrop</code>
         </td >
         <td>
            - update: gt (img)
         </td >
      </tr>
      <tr>
         <td>
            <code>CropLike</code>
         </td >
         <td>
            - update: specified by <code>target_key</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>GetMaskedImage</code>
         </td >
         <td>
            - add: masked_img
         </td >
      </tr>
      <tr>
         <td>
            <code>GenerateFacialHeatmap</code>
         </td >
         <td>
            - add: heatmap
         </td >
      </tr>
      <tr>
         <td>
            <code>GenerateCoordinateAndCell</code>
         </td >
         <td>
            - add: coord, cell
            - update: gt (img)
         </td >
      </tr>
      <tr>
         <td>
            <code>Normalize</code>
         </td >
         <td>
            - add: img_norm_cfg
            - update: specified by <code>keys</code>
         </td >
      </tr>
      <tr>
         <td>
            <code>RescaleToZeroOne</code>
         </td >
         <td>
            - update: specified by <code>keys</code>
         </td >
      </tr>
   </tbody>
</table>

### Formatting

<table class="docutils">
   <thead>
      <tr>
         <th style="text-align:center">Transform</th>
         <th style="text-align:center">Modification of Results' keys</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>
            <code>ToTensor</code>
         </td>
         <td>
            update: specified by <code>keys</code>.
         </td>
      </tr>
      <tr>
         <td>
            <code>FormatTrimap</code>
         </td>
         <td>
            - update: trimap
         </td>
      </tr>
      <tr>
         <td>
            <code>PackInputs</code>
         </td>
         <td>
            - add: inputs, data_sample
            - remove: all other keys
         </td>
      </tr>
   </tbody>
</table>

### Albumentations

MMagic support adding custom transformations from [Albumentations](https://github.com/albumentations-team/albumentations) library. Please visit https://albumentations.ai/docs/getting_started/transforms_and_targets to get more information.

An example of Albumentations's `transforms` is as followed:

```python
albu_transforms = [
   dict(
         type='Resize',
         height=100,
         width=100,
   ),
   dict(
         type='RandomFog',
         p=0.5,
   ),
   dict(
         type='RandomRain',
         p=0.5
   ),
   dict(
         type='RandomSnow',
         p=0.5,
   ),
]
pipeline = [
   dict(
         type='LoadImageFromFile',
         key='img',
         color_type='color',
         channel_order='rgb',
         imdecode_backend='cv2'),
   dict(
         type='Albumentations',
         keys=['img'],
         transforms=albu_transforms),
   dict(type='PackInputs')
]
```

## Extend and use custom pipelines

### A simple example of MyTransform

1. Write a new pipeline in a file, e.g., in `my_pipeline.py`. It takes a dict as input and returns a dict.

```python
import random
from mmcv.transforms import BaseTransform
from mmagic.registry import TRANSFORMS


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
