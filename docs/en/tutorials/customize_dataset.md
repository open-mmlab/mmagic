# Tutorial 1: Customize Datasets

## Supported Data Format

### Image Super-Resolution

- [SRAnnotationDataset](/mmedit/datasets/sr_annotation_dataset.py)
  General paired image dataset with an annotation file for image restoration.
- [SRFolderDataset](/mmedit/datasets/sr_folder_dataset.py)
  General paired image folder dataset for image restoration.
- [SRFolderGTDataset](/mmedit/datasets/sr_folder_gt_dataset.py)
  General ground-truth image folder dataset for image restoration, where low-quality image should be generated in pipeline.
- [SRFolderRefDataset](/mmedit/datasets/sr_folder_ref_dataset.py)
  General paired image folder dataset for reference-based image restoration.
- [SRLmdbDataset](/mmedit/datasets/sr_lmdb_dataset.py)
  General paired image lmdb dataset for image restoration.
- [SRFacialLandmarkDataset](/mmedit/datasets/sr_facial_landmark_dataset.py)
  Facial image and landmark dataset with an annotation file.

### Video Super-Resolution

- [SRFolderMultipleGTDataset](/mmedit/datasets/sr_folder_multiple_gt_dataset.py)
  General dataset for video super resolution, used for recurrent networks.
- [SRREDSDataset](/mmedit/datasets/sr_reds_dataset.py)
  REDS dataset for video super resolution.
- [SRREDSMultipleGTDataset](/mmedit/datasets/sr_reds_multiple_gt_dataset.py)
  REDS dataset for video super resolution for recurrent networks.
- [SRTestMultipleGTDataset](/mmedit/datasets/sr_test_multiple_gt_dataset.py)
  Test dataset for video super resolution for recurrent networks.
- [SRVid4Dataset](/mmedit/datasets/sr_vid4_dataset.py)
  Vid4 dataset for video super resolution.
- [SRVimeo90KDataset](/mmedit/datasets/sr_vimeo90k_dataset.py)
  Vimeo90K dataset for video super resolution.
- [SRVimeo90KMultipleGTDataset](/mmedit/datasets/sr_vimeo90k_multiple_gt_dataset.py)
  Vimeo90K dataset for video super resolution for recurrent networks.

### Video Frame Interpolation

- [VFIVimeo90KDataset](/mmedit/datasets/vfi_vimeo90k_dataset.py)
  Vimeo90K dataset for video frame interpolation.

### Matting

- [AdobeComp1kDataset](/mmedit/datasets/comp1k_dataset.py)
  Adobe composition-1k dataset.

### Inpainting

- [ImgInpaintingDataset](/mmedit/datasets/img_inpainting_dataset.py)
  Only use the image name information from annotation file.

### Generation

- [GenerationPairedDataset](/mmedit/datasets/generation_paired_dataset.py)
  General paired image folder dataset for image generation.
- [GenerationUnpairedDataset](/mmedit/datasets/generation_unpaired_dataset.py)
  General unpaired image folder dataset for image generation.

## Support new data format

You can reorganize new data formats to existing format.

Or create a new dataset in [mmedit/datasets](/mmedit/datasets) to load the data.

Inheriting from the base class of datasets will make it easier to create a new dataset

- [BaseSRDataset](/mmedit/datasets/base_sr_dataset.py)
- [BaseVFIDataset](/mmedit/datasets/base_vfi_dataset.py)
- [BaseMattingDataset](/mmedit/datasets/base_matting_dataset.py)
- [BaseGenerationDataset](/mmedit/datasets/base_generation_dataset.py)

Here is an example of create a dataset for video frame interpolation:

```python
import os
import os.path as osp

from .base_vfi_dataset import BaseVFIDataset
from .registry import DATASETS


@DATASETS.register_module()
class NewVFIDataset(BaseVFIDataset):
    """Introduce the dataset

    Examples of file structure.

    Args:
        pipeline (list[dict | callable]): A sequence of data transformations.
        folder (str | :obj:`Path`): Path to the folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, pipeline, folder, ann_file, test_mode=False):
        super().__init__(pipeline, folder, ann_file, test_mode)
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

If you want create a dataset for a new low level CV task (e.g. denoise, derain, defog, and de-reflection), you can inheriting from [BaseDataset](/mmedit/datasets/base_dataset.py).

Here is an example of create a base dataset for denoising:

```python
import copy
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from .pipelines import Compose

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


class BaseDnDataset(BaseDataset):
    """Base class for denoising datasets.
    """

    # If any extra parameter is required, please rewrite the `__init__`
    # def __init__(self, pipeline, new_para, test_mode=False):
    #     super().__init__(pipeline, test_mode)
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

We use [RepeatDataset](mmedit/datasets/dataset_wrappers.py) as wrapper to repeat the dataset.
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
