# Prepare Your Own Datasets

In this document, we will introduce the design of each datasets in MMEditing and how users can design their own dataset.

- [Supported Dataset](#supported-data-format)
  - [BasicImageDataset](#basicimagedataset)
  - [BasicFramesDataset](#basicframesdataset)
  - [AdobeComp1kDataset](#adobecomp1kdataset)
- [Design a new dataset](#design-a-new-dataset)
  - [Repeat dataset](#repeat-dataset)

## Supported Data Format

In 1.x version of MMEditing, all datasets are inherited from `BaseDataset`.
Each dataset load the list of data info (e.g., data path) by `load_data_list`.
In `__getitem__`, `prepare_data` is called to get the preprocessed data.
In `prepare_data`, data loading pipeline consists of the following steps:

1. fetch the data info by passed index, implemented by `get_data_info`
2. apply data transforms to the data, implemented by `pipeline`

### BasicImageDataset

- [BasicImageDataset](/mmedit/datasets/basic_image_dataset.py)
  General image dataset designed for low-level vision tasks with image, such as image super-resolution and inpainting. The annotation file is optional.

### BasicFramesDataset

- [BasicFramesDataset](/mmedit/datasets/basic_frames_dataset.py)
  General frames dataset designed for low-level vision tasks with frames, such as video super-resolution and video frame interpolation. The annotation file is optional.

### AdobeComp1kDataset

- [AdobeComp1kDataset](/mmedit/datasets/comp1k_dataset.py)
  Adobe composition-1k dataset.

### UnconditionalImageDataset

`UnconditionalImageDataset` is used for loading data for unconditional GAN models (e.g., StyleGANv2, StyleGANv3, WGAN-GP).
In this class, we implement `load_data_list` to scan the data list from passed `data_root` and use the default data loading logic provided by `BaseDataset`.

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

`PairedImageDataset` is designed for translation models that needs paried training data (e.g., Pix2Pix).
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

Or create a new dataset in [mmedit/datasets](/mmedit/datasets) to load the data.

Inheriting from the base class of datasets such as [BasicImageDataset](/mmedit/datasets/basic_image_dataset.py) and [BasicFramesDataset](/mmedit/datasets/basic_frames_dataset.py) will make it easier to create a new dataset.

And you can create a new dataset inherited from [BaseDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py) which is the base class of datasets in [MMEngine](https://github.com/open-mmlab/mmengine).

Here is an example of creating a dataset for video frame interpolation:

```python
from .basic_frames_dataset import BasicFramesDataset
from mmedit.registry import DATASETS


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

Welcome to [submit new dataset classes to MMEditing](https://github.com/open-mmlab/mmediting/compare).

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

You may refer to [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md).
