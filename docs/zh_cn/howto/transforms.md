# 如何设计自己的数据变换

在本教程中，我们将介绍MMagic中变换流水线的设计。

The structure of this guide are as follows:

- [如何设计自己的数据变换](#如何设计自己的数据变换)
  - [MMagic中的数据流水线](#mmagic中的数据流水线)
    - [数据变换的一个简单示例](#数据变换的一个简单示例)
    - [BasicVSR的一个示例](#basicvsr的一个示例)
    - [Pix2Pix的一个示例](#pix2pix的一个示例)
  - [MMagic中支持的数据变换](#mmagic中支持的数据变换)
    - [数据加载](#数据加载)
    - [预处理](#预处理)
    - [格式化](#格式化)
  - [扩展和使用自定义流水线](#扩展和使用自定义流水线)
    - [一个简单的MyTransform示例](#一个简单的mytransform示例)
    - [一个翻转变换的示例](#一个翻转变换的示例)

## MMagic中的数据流水线

按照典型的惯例，我们使用 `Dataset` 和 `DataLoader` 来加载多个线程的数据。 `Dataset` 返回一个与模型的forward方法的参数相对应的数据项的字典。

数据准备流水线和数据集是分开的。通常，一个数据集定义了如何处理标注，而一个数据管道定义了准备一个数据字典的所有步骤。

一个流水线由一连串的操作组成。每个操作都需要一个字典作为输入，并为下一个变换输出一个字典。

这些操作被分为数据加载、预处理和格式化。

在MMagic中，所有数据变换都继承自 `BaseTransform`。
变换的输入和输出类型都是字典。

### 数据变换的一个简单示例

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

一般来说，变换流水线的最后一步必须是 `PackInputs`.
`PackInputs` 将把处理过的数据打包成一个包含两个字段的字典：`inputs` 和 `data_samples`.
`inputs` 是你想用作模型输入的变量，它可以是 `torch.Tensor` 的类型， `torch.Tensor` 的字典，或者你想要的任何类型。
`data_samples` 是一个 `DataSample` 的列表. 每个 `DataSample` 都包含真实值和对应输入的必要信息。

### BasicVSR的一个示例

下面是一个BasicVSR的流水线示例。

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

对于每个操作，我们列出了添加/更新/删除的相关字典字段，标记为 '\*' 的字典字段是可选的。

### Pix2Pix的一个示例

下面是一个在aerial2maps数据集上Pix2Pix训练的流水线示例。

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

## MMagic中支持的数据变换

### 数据加载

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

### 预处理

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

### 格式化

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

MMagic 支持添加 [Albumentations](https://github.com/albumentations-team/albumentations) 库中的 transformation，请浏览 https://albumentations.ai/docs/getting_started/transforms_and_targets 获取更多 transformation 的信息。

使用 Albumentations 的示例如下：

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

## 扩展和使用自定义流水线

### 一个简单的MyTransform示例

1. 在文件中写入一个新的流水线，例如在 `my_pipeline.py`中。它接受一个字典作为输入，并返回一个字典。

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

2. 在你的配置文件中导入并使用该流水线。

确保导入相对于你的训练脚本所在的位置。

```python
train_pipeline = [
    ...
    dict(type='MyTransform', p=0.2),
    ...
]
```

### 一个翻转变换的示例

这里我们以一个简单的翻转变换为例：

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

因此，我们可以实例化一个 `MyFlip` 对象，用它来处理数据字典。

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

或者，我们可以在配置文件的数据流水线中使用 `MyFlip` 变换。

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

请注意，如果你想在配置中使用 `MyFlip` ，你必须确保在程序运行过程中导入包含 `MyFlip` 的文件。
