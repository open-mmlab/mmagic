# Tutorial 1: Customize Datasets

- [Customize Datasets](#Tutorial-1:-Customize-Datasets)
  - [Supported Data Format](#Supported-Data-Format)
  - [Support new data format](#Support-new-data-format)
  - [Customize datasets by dataset wrappers](#Customize-datasets-by-dataset-wrappers)
    - [Repeat dataset](#Repeat-dataset)

We support three types of datasets listed as [Supported Data Format](##supported-data-format): image, frames and Adobe composition-1k dataset. Customized datasets are supposed to inherit from one of them following [Support new data format](##support-new-data-format) tutorial.

## Supported Data Format

- [BasicImageDataset](/mmedit/datasets/basic_image_dataset.py)
  General image dataset designed for low-level vision tasks with image, such as image super-resolution and inpainting. The annotation file is optional.
- [BasicFramesDataset](/mmedit/datasets/basic_frames_dataset.py)
  General frames dataset designed for low-level vision tasks with frames, such as video super-resolution and video frame interpolation. The annotation file is optional.
- [AdobeComp1kDataset](/mmedit/datasets/comp1k_dataset.py)
  Adobe composition-1k dataset.

## Support new data format

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

## Customize datasets by dataset wrappers

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
