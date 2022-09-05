# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

import numpy as np
from mmengine import FileClient
from mmengine.dataset import BaseDataset, force_full_init

from mmedit.registry import DATASETS
from .utils import infer_io_backend

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class UnpairedImageDataset(BaseDataset):
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
        io_backend (str, optional): The storage backend type. Options are
            "disk", "ceph", "memcached", "lmdb", "http" and "petrel".
            Default: None.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        domain_a (str, optional): Domain of images in trainA / testA.
            Defaults to None.
        domain_b (str, optional): Domain of images in trainB / testB.
            Defaults to None.
    """

    def __init__(self,
                 data_root,
                 pipeline,
                 io_backend: Optional[str] = None,
                 test_mode=False,
                 domain_a=None,
                 domain_b=None):
        phase = 'test' if test_mode else 'train'
        self.dataroot_a = osp.join(str(data_root), phase + 'A')
        self.dataroot_b = osp.join(str(data_root), phase + 'B')

        if io_backend is None:
            io_backend = infer_io_backend(data_root)
        self.file_client = FileClient(backend=io_backend)

        super().__init__(
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=False)
        self.len_a = len(self.data_infos_a)
        self.len_b = len(self.data_infos_b)
        self.test_mode = test_mode
        assert isinstance(domain_a, str)
        assert isinstance(domain_b, str)
        self.domain_a = domain_a
        self.domain_b = domain_b

    def load_data_list(self):
        self.data_infos_a = self._load_domain_data_list(self.dataroot_a)
        self.data_infos_b = self._load_domain_data_list(self.dataroot_b)
        return [self.data_infos_a, self.data_infos_b]

    def _load_domain_data_list(self, dataroot):
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

    # def prepare_train_data(self, idx):
    #     """Prepare unpaired training data.

    #     Args:
    #         idx (int): Index of current batch.

    #     Returns:
    #         dict: Prepared training data batch.
    #     """
    #     img_a_path = self.data_infos_a[idx % self.len_a]['path']
    #     idx_b = np.random.randint(0, self.len_b)
    #     img_b_path = self.data_infos_b[idx_b]['path']
    #     results = dict()
    #     results[f'img_{self.domain_a}_path'] = img_a_path
    #     results[f'img_{self.domain_b}_path'] = img_b_path
    #     return self.pipeline(results)

    # def prepare_test_data(self, idx):
    #     """Prepare unpaired test data.

    #     Args:
    #         idx (int): Index of current batch.

    #     Returns:
    #         list[dict]: Prepared test data batch.
    #     """
    #     img_a_path = self.data_infos_a[idx % self.len_a]['path']
    #     img_b_path = self.data_infos_b[idx % self.len_b]['path']
    #     results = dict()
    #     results[f'img_{self.domain_a}_path'] = img_a_path
    #     results[f'img_{self.domain_b}_path'] = img_b_path
    #     return self.pipeline(results)

    @force_full_init
    def get_data_info(self, idx) -> dict:
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        if self.test_mode:
            idx_b = np.random.randint(0, self.len_b)
            img_b_path = self.data_infos_b[idx_b]['path']
        else:
            img_b_path = self.data_infos_b[idx % self.len_b]['path']
        data_info = dict()
        data_info[f'img_{self.domain_a}_path'] = img_a_path
        data_info[f'img_{self.domain_b}_path'] = img_b_path
        return data_info

    def __len__(self):
        return max(self.len_a, self.len_b)

    def scan_folder(self, path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: Image list obtained from the given folder.
        """
        imgs_list = self.file_client.list_dir_or_file(
            path, list_dir=False, suffix=IMG_EXTENSIONS, recursive=True)
        images = [self.file_client.join_path(path, img) for img in imgs_list]
        assert images, f'{path} has no valid image file.'
        return images

    # def __getitem__(self, idx):
    #     """Get item at each call.

    #     Args:
    #         idx (int): Index for getting each item.
    #     """
    #     if not self.test_mode:
    #         return self.prepare_train_data(idx)

    #     return self.prepare_test_data(idx)
