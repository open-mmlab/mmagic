# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Tuple

import torch
from mmengine.dataset import ConcatDataset

from mmedit.registry import DATASETS

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class PairedMultipathDataset(ConcatDataset):
    """General paired image dataset for multiple datasets.

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'.
    '/path/to/data' can be initialized by args 'dataroot'.
    Each sample contains a pair of
    images concatenated in the w dimension (A|B).

    Args:
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        mini_ratio (float, optional): use what ratio of images in each datasets
            Defaults to 1.0
        fix_length (int, optional): use how many images
            for each datasets in each epochs.
            Defaults to None.
            (mini_ratio and fix_length cannot be set at the same time.
            if fix_length is set, only consider fix_length.)
        shuffle (bool, optional): Whether shuffle the order of picking datasets
            If false, it will pick one image squentially from the order of
            datasets configurations.
            Defaults to True.

    Note:
        datasets: [
            dict(
                type='BasicImageDataset',
                data_root='../datasets/gopro/train',
                data_prefix=dict(gt='sharp', img='blur'),
                ann_file='meta_info_gopro_train.txt',
                pipeline=[]),
            dict(
                type='BasicImageDataset',
                data_root='../datasets/SIDD/train',
                data_prefix=dict(gt='gt', img='input'),
                pipeline=[])
        ]
    """

    def __init__(self,
                 datasets=[],
                 lazy_init=False,
                 mini_ratio=1.0,
                 fix_length=None,
                 shuffle=True,
                 **kwards):

        super().__init__(datasets, lazy_init)
        self.mini_ratio = mini_ratio
        self.fix_length = fix_length
        self.shuffle = shuffle

        self.num_datasets = len(datasets)
        self.datasets_order = torch.arange(0, self.num_datasets)
        # the index counter for each datasets,
        # only used for fixed dataset length
        self.data_indices = torch.zeros(self.num_datasets, dtype=torch.int)

    def __len__(self):
        if self.fix_length is not None:
            return self.fix_length * self.num_datasets
        return super().__len__()

    def _get_ori_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``PairedMultipathDataset``.

        Returns:
            Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    f'absolute value of index({idx}) should not exceed dataset'
                    f'length({len(self)}).')
            idx = len(self) + idx

        # if fixed length for each dataset,
        # use data_indices counter to get sample idx
        if self.fix_length is not None:
            return self._get_ori_dataset_idx_from_counter(idx)

        # Get `dataset_idx` to tell idx belongs to which dataset.
        if self.shuffle and idx % self.num_datasets == 0:
            self.datasets_order = torch.randperm(self.num_datasets)
        dataset_idx = self.datasets_order[idx % self.num_datasets]
        # Get the inner index of single dataset.
        dataset_len = self.datasets[dataset_idx].__len__()
        image_idx = (idx // self.num_datasets) % dataset_len
        sample_idx = image_idx

        return dataset_idx, sample_idx

    def _get_ori_dataset_idx_from_counter(self, idx) -> Tuple[int, int]:
        """Convert global idx to local index from counters for each dataset.

        Args:
            idx (int): Global index of ``PairedMultipathDataset``.

        Returns:
            Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """
        # Get `dataset_idx` to tell idx belongs to which dataset.
        if self.shuffle and idx % self.num_datasets == 0:
            self.datasets_order = torch.randperm(self.num_datasets)
        dataset_idx = self.datasets_order[idx % self.num_datasets]
        # Get the inner index of single dataset.
        sample_idx = copy.deepcopy(self.data_indices[dataset_idx])
        self.data_indices[dataset_idx] = (sample_idx + 1) \
            % self.datasets[dataset_idx].__len__()
        return dataset_idx, sample_idx
