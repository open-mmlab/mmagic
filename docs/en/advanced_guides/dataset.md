# Tutorial 1: Customize Datasets

## Supported Data Format

- [BasicImageDataset](/mmedit/datasets/basic_image_dataset.py)
  General image dataset designed for low-level vision tasks with image, such as image super-resolution and inpainting. The annotation file is optional.
- [BasicFramesDataset](/mmedit/datasets/basic_frames_dataset.py)
  General frames dataset designed for low-level vision tasks with frames, such as video super-resolution and video frame interpolation. The annotation file is optional.
- [AdobeComp1kDataset](/mmedit/datasets/comp1k_dataset.py)
  Adobe composition-1k dataset.

## Support new data format

You can reorganize new data formats to existing format.

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

If you want create a dataset for a new low level CV task (e.g. denoise, derain, defog, and de-reflection), you can inherit from [BasicImageDataset](/mmedit/datasets/basic_image_dataset.py) or [BasicFramesDataset](/mmedit/datasets/basic_frames_dataset.py).

Here is an example of creating a base dataset for denoising:

```python
import copy

from .basic_image_dataset import BasicImageDataset

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


class BaseDnDataset(BasicImageDataset):
    """Base class for denoising datasets.
    """

    # If any extra parameter is required, please rewrite the `__init__`
    # def __init__(self, old_para, new_para, test_mode=False):
    #     super().__init__(old_para, test_mode)
    #     self.new_para = new_para

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: image list obtained form given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = list(scandir(path, suffix=IMG_EXTENSIONS, recursive=True))
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.

        Returns:
            dict: The output dict of pipeline.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
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
