# How to prepare your own datasets

In this document, we will introduce the design of each datasets in MMagic and how users can design their own dataset.

- [How to prepare your own datasets](#how-to-prepare-your-own-datasets)
  - [Supported Data Format](#supported-data-format)
    - [BasicImageDataset](#basicimagedataset)
    - [BasicFramesDataset](#basicframesdataset)
    - [BasicConditonalDataset](#basicconditonaldataset)
      - [1. Annotation file read by line (e.g., txt)](#1-annotation-file-read-by-line-eg-txt)
      - [2. Dict-based annotation file (e.g., json):](#2-dict-based-annotation-file-eg-json)
      - [3. Folder-based annotation (no annotation file need):](#3-folder-based-annotation-no-annotation-file-need)
    - [ImageNet Dataset and CIFAR10 Dataset](#imagenet-dataset-and-cifar10-dataset)
    - [AdobeComp1kDataset](#adobecomp1kdataset)
    - [GrowScaleImgDataset](#growscaleimgdataset)
    - [SinGANDataset](#singandataset)
    - [PairedImageDataset](#pairedimagedataset)
    - [UnpairedImageDataset](#unpairedimagedataset)
  - [Design a new dataset](#design-a-new-dataset)
    - [Repeat dataset](#repeat-dataset)

## Supported Data Format

In MMagic, all datasets are inherited from `BaseDataset`.
Each dataset load the list of data info (e.g., data path) by `load_data_list`.
In `__getitem__`, `prepare_data` is called to get the preprocessed data.
In `prepare_data`, data loading pipeline consists of the following steps:

1. fetch the data info by passed index, implemented by `get_data_info`
2. apply data transforms to the data, implemented by `pipeline`

### BasicImageDataset

**BasicImageDataset** `mmagic.datasets.BasicImageDataset`
General image dataset designed for low-level vision tasks with image, such as image super-resolution, inpainting and unconditional image generation. The annotation file is optional.

If use annotation file, the annotation format can be shown as follows.

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

Here we give several examples showing how to use `BasicImageDataset`. Assume the file structure as the following:

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

Case 1: Loading DIV2K dataset for training a SISR model.

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

Case 2: Loading places dataset for training an inpainting model.

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

Case 3: Loading CelebA-HQ dataset for training an PGGAN.

```python
dataset = BasicImageDataset(
        pipeline=[],
        data_root='./data/celebahq/imgs_1024')
```

### BasicFramesDataset

**BasicFramesDataset** `mmagic.datasets.BasicFramesDataset`
General frames dataset designed for low-level vision tasks with frames, such as video super-resolution and video frame interpolation. The annotation file is optional.

If use annotation file, the annotation format can be shown as follows.

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

Assume the file structure as the following:

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

Case 1: Loading Vid4 dataset for training a VSR model.

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

Case 2: Loading Vimeo90k dataset for training a VFI model.

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

**BasicConditonalDataset** `mmagic.datasets.BasicConditonalDataset` is designed for conditional GANs (e.g., SAGAN, BigGAN). This dataset support load label for the annotation file. `BasicConditonalDataset` support three kinds of annotation as follow:

#### 1. Annotation file read by line (e.g., txt)

Sample files structure:

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

Sample annotation file (the first column is the image path and the second column is the index of category):

```
    folder_1/xxx.png 0
    folder_1/xxy.png 1
    folder_2/123.png 5
    folder_2/nsdf3.png 3
    ...
```

Config example for ImageNet dataset:

```python
dataset=dict(
    type='BasicConditionalDataset,
    data_root='./data/imagenet/',
    ann_file='meta/train.txt',
    data_prefix='train',
    pipeline=train_pipeline),
```

#### 2. Dict-based annotation file (e.g., json):

Sample files structure:

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

Sample annotation file (the key is the image path and the value column
is the label):

```
    {
        "folder_1/xxx.png": [1, 2, 3, 4],
        "folder_1/xxy.png": [2, 4, 1, 0],
        "folder_2/123.png": [0, 9, 8, 1],
        "folder_2/nsdf3.png", [1, 0, 0, 2],
        ...
    }
```

Config example for EG3D (shapenet-car) dataset:

```python
dataset = dict(
    type='BasicConditionalDataset',
    data_root='./data/eg3d/shapenet-car',
    ann_file='annotation.json',
    pipeline=train_pipeline)
```

In this kind of annotation, labels can be any type and not restricted to an index.

#### 3. Folder-based annotation (no annotation file need):

Sample files structure:

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

If the annotation file is specified, the dataset will be generated by the first two ways, otherwise, try the third way.

### ImageNet Dataset and CIFAR10 Dataset

**ImageNet Dataset** `mmagic.datasets.ImageNet` and **CIFAR10 Dataset**`mmagic.datasets.CIFAR10` are datasets specific designed for ImageNet and CIFAR10 datasets. Both two datasets are encapsulation of `BasicConditionalDataset`. You can used them to load data from ImageNet dataset and CIFAR10 dataset easily.

Config example for ImageNet:

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

Config example for CIFAR10:

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

**AdobeComp1kDataset** `mmagic.datasets.AdobeComp1kDataset`
Adobe composition-1k dataset.

The dataset loads (alpha, fg, bg) data and apply specified transforms to
the data. You could specify whether composite merged image online or load
composited merged image in pipeline.

Example for online comp-1k dataset:

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

Example for offline comp-1k dataset:

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

`GrowScaleImgDataset` is designed for dynamic GAN models (e.g., PGGAN and StyleGANv1).
In this dataset, we support switching the data root during training to load training images of different resolutions.
This procedure is implemented by `GrowScaleImgDataset.update_annotations` and is called by `PGGANFetchDataHook.before_train_iter` in the training process.

```python
def update_annotations(self, curr_scale):
    # determine if the data root needs to be updated
    if curr_scale == self._actual_curr_scale:
        return False

    # fetch new data root by resolution (scale)
    for scale in self._img_scales:
        if curr_scale <= scale:
            self._curr_scale = scale
            break
        if scale == self._img_scales[-1]:
            assert RuntimeError(
                f'Cannot find a suitable scale for {curr_scale}')
    self._actual_curr_scale = curr_scale
    self.data_root = self.data_roots[str(self._curr_scale)]

    # reload the data list with new data root
    self.load_data_list()

    # print basic dataset information to check the validity
    print_log('Update Dataset: ' + repr(self), 'current')
    return True
```

### SinGANDataset

`SinGANDataset` is designed for SinGAN's training.
In SinGAN's training, we do not iterate the images in the dataset but return a consistent preprocessed image dict.

Therefore, we bypass the default data loading logic of `BaseDataset` because we do not need to load the corresponding image data based on the given index.

```python
def load_data_list(self, min_size, max_size, scale_factor_init):
    # load single image
    real = mmcv.imread(self.data_root)
    self.reals, self.scale_factor, self.stop_scale = create_real_pyramid(
        real, min_size, max_size, scale_factor_init)

    self.data_dict = {}

    # generate multi scale image
    for i, real in enumerate(self.reals):
        self.data_dict[f'real_scale{i}'] = real

    self.data_dict['input_sample'] = np.zeros_like(
        self.data_dict['real_scale0']).astype(np.float32)

def __getitem__(self, index):
    # directly return the transformed data dict
    return self.pipeline(self.data_dict)
```

### PairedImageDataset

`PairedImageDataset` is designed for translation models that needs paired training data (e.g., Pix2Pix).
The directory structure is shown below. Each image files are the concatenation of the image pair.

```
./data/dataset_name/
├── test
│   └── XXX.jpg
└── train
    └── XXX.jpg
```

In `PairedImageDataset`, we scan the file list in `load_data_list` and save path in `pair_path` field to fit the `LoadPairedImageFromFile` transformation.

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

`UnpairedImageDataset` is designed for translation models that do not need paired data (e.g., CycleGAN). The directory structure is shown below.

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

In this dataset, we overwrite `__getitem__` function to load random image pair in the training process.

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

## Design a new dataset

If you want to create a dataset for a new low level CV task (e.g. denoise, derain, defog, and de-reflection) or existing dataset format doesn't meet your need, you can reorganize new data formats to existing format.

Or create a new dataset in `mmagic/datasets` to load the data.

Inheriting from the base class of datasets such as `BasicImageDataset` and `BasicFramesDataset` will make it easier to create a new dataset.

And you can create a new dataset inherited from [BaseDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py) which is the base class of datasets in [MMEngine](https://github.com/open-mmlab/mmengine).

Here is an example of creating a dataset for video frame interpolation:

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

Welcome to [submit new dataset classes to MMagic](https://github.com/open-mmlab/mmagic/compare).

### Repeat dataset

We use [RepeatDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/dataset_wrapper.py) as wrapper to repeat the dataset.
For example, suppose the original dataset is Dataset_A, to repeat it, the config looks like the following

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

You may refer to [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/basedataset.md).
