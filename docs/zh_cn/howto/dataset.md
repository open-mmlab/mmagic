# 如何自定义数据集

本文档将介绍 MMagic 中每一个数据集的设计方式，以及用户如何设计自定义数据集。

- [如何自定义数据集](#如何自定义数据集)
  - [支持的数据集格式](#支持的数据集格式)
    - [BasicImageDataset](#basicimagedataset)
    - [BasicFramesDataset](#basicframesdataset)
    - [BasicConditonalDataset](#basicconditonaldataset)
      - [1. 逐行读取的标注文件格式（例如 txt 文件）](#1-逐行读取的标注文件格式例如-txt-文件)
      - [2. 基于字典的标注文件格式（例如 json）](#2-基于字典的标注文件格式例如-json)
      - [3. 基于文件夹的标注格式（无需标注文件）](#3-基于文件夹的标注格式无需标注文件)
    - [ImageNet 和 CIFAR10 数据集](#imagenet-和-cifar10-数据集)
    - [AdobeComp1kDataset](#adobecomp1kdataset)
    - [GrowScaleImgDataset](#growscaleimgdataset)
    - [SinGANDataset](#singandataset)
    - [PairedImageDataset](#pairedimagedataset)
    - [UnpairedImageDataset](#unpairedimagedataset)
  - [实现一个新的数据集](#实现一个新的数据集)
    - [重复数据集](#重复数据集)

## 支持的数据集格式

在 MMagic 中，所有的数据集都是从 `BaseDataset` 类继承而来的。
每个数据集都通过 `load_data_list` 方法来加载数据信息列表（例如数据所在的路径）。
在 `__getitem__` 方法中，调用 `prepare_data` 来获取前处理后的数据。
在 `prepare_data` 方法中，数据加载流程包括如下步骤：

1. 通过传入的索引来获取数据信息，由 `get_data_info` 方法实现。
2. 对数据应用数据转换，由 `pipeline` 方法实现。

### BasicImageDataset

**BasicImageDataset** `mmagic.datasets.BasicImageDataset` 是一个通用图片数据集，是为了底层视觉任务而设计的，比如图像超分辨率，图像修复和无条件图像生成。可以选择是否使用标注文件。

如使用标注文件，标注的格式可以如下所示：

```bash
   Case 1 (CelebA-HQ):

       000001.png
       000002.png

   Case 2 (DIV2K):

       0001_s001.png (480,480,3)
       0001_s002.png (480,480,3)
       0001_s003.png (480,480,3)
       0002_s001.png (480,480,3)
       0002_s002.png (480,480,3)

   Case 3 (Vimeo90k):

       00001/0266 (256, 448, 3)
       00001/0268 (256, 448, 3)
```

下面我们给出几个如何使用 `BasicImageDataset` 的示例。假定文件结构如下：

```md
mmagic (root)
├── mmagic
├── tools
├── configs
├── data
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │   │   ├── image.png
│   │   ├── DIV2K_train_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   │   │   │   ├── image_x4.png
│   │   ├── DIV2K_valid_HR
│   │   ├── DIV2K_valid_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   ├── places
│   │   ├── test_set
│   │   ├── train_set
|   |   ├── meta
|   |   |    ├── Places365_train.txt
|   |   |    ├── Places365_val.txt
|   ├── celebahq
│   │   ├── imgs_1024

```

按照以上的文件结构给出 3 个示例。

示例 1: 加载 `DIV2K` 数据集来训练一个 `SISR` 模型。

```python
   dataset = BasicImageDataset(
       ann_file='',
       metainfo=dict(
           dataset_type='div2k',
           task_name='sisr'),
       data_root='data/DIV2K',
       data_prefix=dict(
           gt='DIV2K_train_HR', img='DIV2K_train_LR_bicubic/X4'),
       filename_tmpl=dict(img='{}_x4', gt='{}'),
       pipeline=[])
```

示例 2: 加载 `places` 数据集来训练一个 `inpainting` 模型.

```python
   dataset = BasicImageDataset(
       ann_file='meta/Places365_train.txt',
       metainfo=dict(
           dataset_type='places365',
           task_name='inpainting'),
       data_root='data/places',
       data_prefix=dict(gt='train_set'),
       pipeline=[])
```

示例 3: 加载 `CelebA-HQ` 数据集来训练一个 `PGGAN` 模型.

```python
dataset = BasicImageDataset(
        pipeline=[],
        data_root='./data/celebahq/imgs_1024')
```

### BasicFramesDataset

**BasicFramesDataset** `mmagic.datasets.BasicFramesDataset` 也是一个通用图片数据集，为视频帧的底层视觉任务而设计的，比如视频的超分辨率和视频帧插值。
可以选择是否使用标注文件。

如使用标注文件, 标注的格式示例所示：

```bash
Case 1 (Vid4):

   calendar 41
   city 34
   foliage 49
   walk 47

Case 2 (REDS):

   000/00000000.png (720, 1280, 3)
   000/00000001.png (720, 1280, 3)

Case 3 (Vimeo90k):

   00001/0266 (256, 448, 3)
   00001/0268 (256, 448, 3)
```

假定文件结构如下：

```bash
mmagic (root)
├── mmagic
├── tools
├── configs
├── data
│   ├── Vid4
│   │   ├── BIx4
│   │   │   ├── city
│   │   │   │   ├── img1.png
│   │   ├── GT
│   │   │   ├── city
│   │   │   │   ├── img1.png
│   │   ├── meta_info_Vid4_GT.txt
│   ├── vimeo-triplet
│   │   ├── sequences
|   |   |   ├── 00001
│   │   │   │   ├── 0389
│   │   │   │   │   ├── img1.png
│   │   │   │   │   ├── img2.png
│   │   │   │   │   ├── img3.png
│   │   ├── tri_trainlist.txt
```

按照以上的文件结构给出两个示例。

示例 1: 加载 `Vid4` 数据集来训练一个 `VSR` 模型.

```python
dataset = BasicFramesDataset(
    ann_file='meta_info_Vid4_GT.txt',
    metainfo=dict(dataset_type='vid4', task_name='vsr'),
    data_root='data/Vid4',
    data_prefix=dict(img='BIx4', gt='GT'),
    pipeline=[],
    depth=2,
    num_input_frames=5)
```

示例 2: 加载 `Vimeo90k` 数据集来训练一个 `VFI` 模型.

```python
dataset = BasicFramesDataset(
    ann_file='tri_trainlist.txt',
    metainfo=dict(dataset_type='vimeo90k', task_name='vfi'),
    data_root='data/vimeo-triplet',
    data_prefix=dict(img='sequences', gt='sequences'),
    pipeline=[],
    depth=2,
    load_frames_list=dict(
        img=['img1.png', 'img3.png'], gt=['img2.png']))
```

### BasicConditonalDataset

**BasicConditonalDataset** `mmagic.datasets.BasicConditonalDataset` 是为条件生成对抗网络而设计的（例如 `SAGAN`、`BigGAN`）。该数据集支持为标注文件加载标签。 `BasicConditonalDataset` 支持如下 3 种标注格式。

#### 1. 逐行读取的标注文件格式（例如 txt 文件）

样本文件结构：

```
    data_prefix/
    ├── folder_1
    │   ├── xxx.png
    │   ├── xxy.png
    │   └── ...
    └── folder_2
        ├── 123.png
        ├── nsdf3.png
        └── ...
```

样本标注文件格式（第一列是图像的路径，第二列是类别的索引）

```
    folder_1/xxx.png 0
    folder_1/xxy.png 1
    folder_2/123.png 5
    folder_2/nsdf3.png 3
    ...
```

`ImageNet` 数据集的配置示例：

```python
dataset=dict(
    type='BasicConditionalDataset,
    data_root='./data/imagenet/',
    ann_file='meta/train.txt',
    data_prefix='train',
    pipeline=train_pipeline),
```

#### 2. 基于字典的标注文件格式（例如 json）

样本文件结构：

```
    data_prefix/
    ├── folder_1
    │   ├── xxx.png
    │   ├── xxy.png
    │   └── ...
    └── folder_2
        ├── 123.png
        ├── nsdf3.png
        └── ...
```

样本标注文件格式（键为图像的路径，值为标签）。

```
    {
        "folder_1/xxx.png": [1, 2, 3, 4],
        "folder_1/xxy.png": [2, 4, 1, 0],
        "folder_2/123.png": [0, 9, 8, 1],
        "folder_2/nsdf3.png", [1, 0, 0, 2],
        ...
    }
```

`EG3D (shapenet-car) ` 数据集的配置示例：

```python
dataset = dict(
    type='BasicConditionalDataset',
    data_root='./data/eg3d/shapenet-car',
    ann_file='annotation.json',
    pipeline=train_pipeline)
```

在这种类型的注释中，标签可以是任何类型，不仅限于索引。

#### 3. 基于文件夹的标注格式（无需标注文件）

样本文件结构：

```
    data_prefix/
    ├── class_x
    │   ├── xxx.png
    │   ├── xxy.png
    │   └── ...
    │       └── xxz.png
    └── class_y
        ├── 123.png
        ├── nsdf3.png
        ├── ...
        └── asd932_.png
```

如果在配置的 `ann_file` 中指定了标注文件，则将使用上面的前两种方式生成数据集，否则将尝试使用第三种方式。

### ImageNet 和 CIFAR10 数据集

**ImageNet 数据集** `mmagic.datasets.ImageNet` 和 **CIFAR10 数据集**`mmagic.datasets.CIFAR10` 是为 `ImageNet` 和 `CIFAR10` 这两个数据集而设计的。
这两个数据集都是基于 `BasicConditionalDataset` 封装的。您可以使用它们来轻松加载这两个数据集的数据。

`ImageNet` 的配置示例：

```python
pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='RandomCropLongEdge', keys=['img']),
    dict(type='Resize', scale=(128, 128), keys=['img'], backend='pillow'),
    dict(type='Flip', keys=['img'], flip_ratio=0.5, direction='horizontal'),
    dict(type='PackInputs')
]

dataset=dict(
    type='ImageNet',
    data_root='./data/imagenet/',
    ann_file='meta/train.txt',
    data_prefix='train',
    pipeline=pipeline),
```

`CIFAR10` 的配置示例:

```python
pipeline = [dict(type='PackInputs')]

dataset = dict(
    type='CIFAR10',
    data_root='./data',
    data_prefix='cifar10',
    test_mode=False,
    pipeline=pipeline)
```

### AdobeComp1kDataset

**AdobeComp1kDataset** `mmagic.datasets.AdobeComp1kDataset` 是为 `Adobe composition-1k` 数据集而设计的。

该数据集加载（alpha, fg, bg）数据，并对数据执行指定的变换。您可以在 `pipeline` 中指定在线合成图像或加载离线已合成的图像。

在线合成 `comp-1k` 数据集示例:

```md
[
   {
       "alpha": 'alpha/000.png',
       "fg": 'fg/000.png',
       "bg": 'bg/000.png'
   },
   {
       "alpha": 'alpha/001.png',
       "fg": 'fg/001.png',
       "bg": 'bg/001.png'
   },
]
```

离线合成 `comp-1k` 数据集示例:

```md
[
  {
      "alpha": 'alpha/000.png',
      "merged": 'merged/000.png',
      "fg": 'fg/000.png',
      "bg": 'bg/000.png'
  },
  {
      "alpha": 'alpha/001.png',
      "merged": 'merged/001.png',
      "fg": 'fg/001.png',
      "bg": 'bg/001.png'
  },
]
```

### GrowScaleImgDataset

`GrowScaleImgDataset` 是为了动态 GAN 模型（例如 `PGGAN` 和 `StyleGANv1`）而设计的。
在这个数据集中，我们支持在训练过程中切换数据根目录，来加载不同分辨率的训练图像。
这个过程是通过 `GrowScaleImgDataset.update_annotations` 方法实现的，并在训练过程中由 `PGGANFetchDataHook.before_train_iter` 调用。

```python
def update_annotations(self, curr_scale):
    # 确定是否需要更新数据根目录
    if curr_scale == self._actual_curr_scale:
        return False

    # 按图像分辨率（尺度）提取新的数据根目录
    for scale in self._img_scales:
        if curr_scale <= scale:
            self._curr_scale = scale
            break
        if scale == self._img_scales[-1]:
            assert RuntimeError(
                f'Cannot find a suitable scale for {curr_scale}')
    self._actual_curr_scale = curr_scale
    self.data_root = self.data_roots[str(self._curr_scale)]

    # 使用新的数据根目录重新加载数据列表
    self.load_data_list()

    # print basic dataset information to check the validity
    print_log('Update Dataset: ' + repr(self), 'current')
    return True
```

### SinGANDataset

`SinGANDataset` 是为 `SinGAN` 模型训练而设计的数据集。在 `SinGAN` 的训练中，我们不会去迭代数据集中的图像，而是返回一个一致的预处理图像字典。
由于不需要根据给定的索引加载相应的图像数据，我们绕过了 `BaseDataset` 的默认数据加载逻辑。

```python
def load_data_list(self, min_size, max_size, scale_factor_init):
    # 加载单张图像
    real = mmcv.imread(self.data_root)
    self.reals, self.scale_factor, self.stop_scale = create_real_pyramid(
        real, min_size, max_size, scale_factor_init)

    self.data_dict = {}

    # 生成多尺度图像
    for i, real in enumerate(self.reals):
        self.data_dict[f'real_scale{i}'] = real

    self.data_dict['input_sample'] = np.zeros_like(
        self.data_dict['real_scale0']).astype(np.float32)

def __getitem__(self, index):
    # 直接返回转换过的数据字典
    return self.pipeline(self.data_dict)
```

### PairedImageDataset

`PairedImageDataset` 专为需要成对训练数据的图像转换模型（例如 `Pix2Pix`）设计。

目录结构如下所示，其中每个图像文件都是图像对的拼接。

```
./data/dataset_name/
├── test
│   └── XXX.jpg
└── train
    └── XXX.jpg
```

在 `PairedImageDataset` 中，我们在 `load_data_list` 方法中扫描文件列表，然后将路径保存在 `pair_path` 字段中，以适配 `LoadPairedImageFromFile` 中的转换。

```python
def load_data_list(self):
    data_infos = []
    pair_paths = sorted(self.scan_folder(self.data_root))
    for pair_path in pair_paths:
        # save path in the specific field
        data_infos.append(dict(pair_path=pair_path))

    return data_infos
```

### UnpairedImageDataset

`UnpairedImageDataset` 是专为不需要成对数据的图像转换模型（例如 `CycleGAN`）设计的数据集。

目录结构如下所示：

```
./data/dataset_name/
├── testA
│   └── XXX.jpg
├── testB
│   └── XXX.jpg
├── trainA
│   └── XXX.jpg
└── trainB
    └── XXX.jpg

```

在该数据集中，我们重载了 `__getitem__` 方法，实现了在训练过程中加载随机的图像对。

```python
def __getitem__(self, idx):
    if not self.test_mode:
        return self.prepare_train_data(idx)

    return self.prepare_test_data(idx)

def prepare_train_data(self, idx):
    img_a_path = self.data_infos_a[idx % self.len_a]['path']
    idx_b = np.random.randint(0, self.len_b)
    img_b_path = self.data_infos_b[idx_b]['path']
    results = dict()
    results[f'img_{self.domain_a}_path'] = img_a_path
    results[f'img_{self.domain_b}_path'] = img_b_path
    return self.pipeline(results)

def prepare_test_data(self, idx):
    img_a_path = self.data_infos_a[idx % self.len_a]['path']
    img_b_path = self.data_infos_b[idx % self.len_b]['path']
    results = dict()
    results[f'img_{self.domain_a}_path'] = img_a_path
    results[f'img_{self.domain_b}_path'] = img_b_path
    return self.pipeline(results)
```

## 实现一个新的数据集

如果您需要为一个新的底层 CV 任务（例如去噪、去雨、去雾和去反射）创建一个数据集，或者现有的数据集格式不符合您的需求，您可以将新的数据格式重新组织成现有的格式，或者在 `mmagic/datasets` 中创建一个新的数据集中来加载数据。

从现有的数据集基类中继承（例如 `BasicImageDataset` 和 `BasicFramesDataset`）会比较容易创建一个新的数据集。

您也可以创建一个继承自 [BaseDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py) 的新数据集，它是定义在 [MMEngine](https://github.com/open-mmlab/mmengine) 中的数据集基类。

下面是创建一个用于视频帧插值的数据集的示例：

```python
from .basic_frames_dataset import BasicFramesDataset
from mmagic.registry import DATASETS


@DATASETS.register_module()
class NewVFIDataset(BasicFramesDataset):
    """Introduce the dataset

    Examples of file structure.

    Args:
        pipeline (list[dict | callable]): A sequence of data transformations.
        folder (str | :obj:`Path`): Path to the folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, ann_file, metainfo, data_root, data_prefix,
                    pipeline, test_mode=False):
        super().__init__(ann_file, metainfo, data_root, data_prefix,
                            pipeline, test_mode)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for the dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        data_infos = []
        ...
        return data_infos

```

欢迎[提交新的数据集类到 MMagic](https://github.com/open-mmlab/mmagic/compare)

### 重复数据集

我们使用 [RepeatDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/dataset_wrapper.py) 作为包装器来重复数据集。
例如，假设原始数据集是 `Dataset_A`，为了重复它，配置文件应该如下所示：

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

您可以参考 [MMEngine 中的教程](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/basedataset.md)。
