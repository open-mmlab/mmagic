# Design Your Own Data Pipelines

- [Customize Data Pipelines](#design-your-own-data-pipelines)
  - [Design of Data pipelines](#design-of-data-pipelines)
    - [Data loading](#data-loading)
    - [Pre-processing](#pre-processing)
    - [Formatting](#formatting)
  - [Extend and use custom pipelines](#extend-and-use-custom-pipelines)

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading with multiple workers. `Dataset` returns a dict of data items corresponding the arguments of models' forward method.

The data preparation pipeline and the dataset is decomposed. Usually a dataset defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.

A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

The operations are categorized into data loading, pre-processing, and formatting

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
