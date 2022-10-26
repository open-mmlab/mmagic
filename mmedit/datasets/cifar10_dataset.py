# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from typing import List, Optional

import mmengine.dist as dist
import numpy as np
from mmengine import FileClient

from mmedit.registry import DATASETS
from .basic_conditional_dataset import BasicConditionalDataset
from .categories import CIFAR10_CATEGORIES
from .data_utils import check_md5, download_and_extract_archive


@DATASETS.register_module()
class CIFAR10(BasicConditionalDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py

    Args:
        data_prefix (str): Prefix for data.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        data_root (str): The root directory for ``data_prefix``.
            Defaults to ''.
        download (bool): Whether to download the dataset if not exists.
            Defaults to True.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """  # noqa: E501

    base_folder = 'cifar-10-batches-py'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    METAINFO = {'classes': CIFAR10_CATEGORIES}

    def __init__(self,
                 data_prefix: str,
                 test_mode: bool,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 download: bool = True,
                 **kwargs):
        self.download = download
        super().__init__(
            # The CIFAR dataset doesn't need specify annotation file
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(root=data_prefix),
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""
        root_prefix = self.data_prefix['root']
        file_client = FileClient.infer_client(uri=root_prefix)

        if dist.is_main_process() and not self._check_integrity():
            if file_client.name != 'HardDiskBackend':
                raise RuntimeError(
                    f'The dataset on {root_prefix} is not integrated, '
                    f'please manually handle it.')

            if self.download:
                download_and_extract_archive(
                    self.url,
                    root_prefix,
                    filename=self.filename,
                    md5=self.tgz_md5)
            else:
                raise RuntimeError(
                    f'Cannot find {self.__class__.__name__} dataset in '
                    f"{self.data_prefix['root']}, you can specify "
                    '`download=True` to download automatically.')

        dist.barrier()
        assert self._check_integrity(), \
            'Download failed or shared storage is unavailable. Please ' \
            f'download the dataset manually through {self.url}.'

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        imgs = []
        gt_labels = []

        # load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = file_client.join_path(root_prefix, self.base_folder,
                                              file_name)
            content = file_client.get(file_path)
            entry = pickle.loads(content, encoding='latin1')
            imgs.append(entry['data'])
            if 'labels' in entry:
                gt_labels.extend(entry['labels'])
            else:
                gt_labels.extend(entry['fine_labels'])

        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
        imgs = imgs.transpose((0, 2, 3, 1))  # convert to HWC

        if self.CLASSES is None:
            # The metainfo in the file has the lowest priority, therefore
            # we only need to load it if classes is not specified.
            self._load_meta()

        data_list = []
        for img, gt_label in zip(imgs, gt_labels):
            info = {'img': img, 'gt_label': int(gt_label)}
            data_list.append(info)
        return data_list

    def _load_meta(self):
        """Load categories information from metafile."""
        root = self.data_prefix['root']
        file_client = FileClient.infer_client(uri=root)

        path = file_client.join_path(root, self.base_folder,
                                     self.meta['filename'])
        md5 = self.meta.get('md5', None)
        if not file_client.exists(path) or (md5 is not None
                                            and not check_md5(path, md5)):
            raise RuntimeError(
                'Dataset metadata file not found or corrupted.' +
                ' You can use `download=True` to download it')
        content = file_client.get(path)
        data = pickle.loads(content, encoding='latin1')
        self._metainfo.setdefault('classes', data[self.meta['key']])

    def _check_integrity(self):
        """Check the integrity of data files."""
        root = self.data_prefix['root']
        file_client = FileClient.infer_client(uri=root)

        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = file_client.join_path(root, self.base_folder, filename)
            if not file_client.exists(fpath):
                return False
            if md5 is not None and not check_md5(
                    fpath, md5, file_client=file_client):
                return False
        return True

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [f"Prefix of data: \t{self.data_prefix['root']}"]
        return body
