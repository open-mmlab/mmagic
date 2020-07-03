import os.path as osp

import numpy as np

from .base_generation_dataset import BaseGenerationDataset
from .registry import DATASETS


@DATASETS.register_module()
class GenerationUnpairedDataset(BaseGenerationDataset):
    """General unpaired image folder dataset for image generation.

    It assumes that the training directory of images from domain A is
    '/path/to/data/trainA', and that from domain B is '/path/to/data/trainB',
    respectively. '/path/to/data' can be initialized by args 'dataroot'.
    During test time, the directory is '/path/to/data/testA' and
    '/path/to/data/testB', respectively.

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of unpaired
            images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, dataroot, pipeline, test_mode=False):
        super(GenerationUnpairedDataset, self).__init__(pipeline, test_mode)
        phase = 'test' if test_mode else 'train'
        self.dataroot_a = osp.join(str(dataroot), phase + 'A')
        self.dataroot_b = osp.join(str(dataroot), phase + 'B')
        self.data_infos_a = self.load_annotations(self.dataroot_a)
        self.data_infos_b = self.load_annotations(self.dataroot_b)
        self.len_a = len(self.data_infos_a)
        self.len_b = len(self.data_infos_b)

    def load_annotations(self, dataroot):
        """Load unpaired image paths of one domain.

        Args:
            dataroot (str): Path to the folder root for unpaired images of
                one domain.

        Returns:
            list[dict]: List that contains unpaired image paths of one domain.
        """
        data_infos = []
        paths = sorted(self.scan_folder(dataroot))
        for path in paths:
            data_infos.append(dict(path=path))
        return data_infos

    def prepare_train_data(self, idx):
        """Prepare unpaired training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        idx_b = np.random.randint(0, self.len_b)
        img_b_path = self.data_infos_b[idx_b]['path']
        results = dict(img_a_path=img_a_path, img_b_path=img_b_path)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare unpaired test data.

        Args:
            idx (int): Index of current batch.

        Returns:
            list[dict]: Prepared test data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        img_b_path = self.data_infos_b[idx % self.len_b]['path']
        results = dict(img_a_path=img_a_path, img_b_path=img_b_path)
        return self.pipeline(results)

    def __len__(self):
        return max(self.len_a, self.len_b)
