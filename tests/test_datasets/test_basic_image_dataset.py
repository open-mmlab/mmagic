# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

import mmcv

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import LoadImageFromFile


class TestImageDatasets:

    @classmethod
    def setup_class(cls):
        cls.data_root = Path(__file__).parent.parent / 'data' / 'image'

    def test_version_1_method(self):

        # test SRAnnotationDataset
        dataset = BasicImageDataset(
            ann_file='train.txt',
            metainfo=dict(
                dataset_type='sr_annotation_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt'),
            filename_tmpl=dict(img='{}_x4'),
            backend_args=dict(backend='local'),
            pipeline=[])
        assert dataset[0] == dict(
            key='baboon',
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

        # test SRFolderDataset
        dataset = BasicImageDataset(
            metainfo=dict(dataset_type='sr_folder_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt'),
            filename_tmpl=dict(img='{}_x4'),
            pipeline=[])
        assert dataset[0] == dict(
            key='baboon',
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

        # test SRFolderGTDataset
        dataset = BasicImageDataset(
            metainfo=dict(dataset_type='sr_folder_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(gt='gt'),
            filename_tmpl=dict(),
            pipeline=[])
        assert dataset[0] == dict(
            key='baboon',
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

        # test SRLmdbDataset
        # The reconstructed LoadImageFromFile supports process images in LMDB
        # backend, which require similar img_path as that of disk backend.
        # Thus SRLmdbDataset is useless.
        # We can build the dataset in the same way with SRAnnotationDataset:
        pipeline = [
            LoadImageFromFile(
                key='img',
                backend_args=dict(
                    backend='lmdb',
                    db_path=Path(__file__).parent.parent / 'data' / 'lq.lmdb'))
        ]
        dataset = BasicImageDataset(
            ann_file=f'lq.lmdb{os.sep}meta_info.txt',
            metainfo=dict(
                dataset_type='sr_annotation_dataset', task_name='sisr'),
            data_prefix=dict(gt='', img=''),
            data_root=self.data_root.parent,
            pipeline=pipeline)
        assert dataset.ann_file == str(self.data_root.parent / 'lq.lmdb' /
                                       'meta_info.txt')
        path_baboon_x4 = Path(
            __file__).parent.parent / 'data' / 'image' / 'lq' / 'baboon_x4.png'
        img_baboon_x4 = mmcv.imread(str(path_baboon_x4), flag='color')
        h, w, _ = img_baboon_x4.shape
        assert dataset[0]['img'].shape == (h, w, 3)
        assert dataset[0]['ori_img_shape'] == (h, w, 3)
        dataset[0]['img_path'] == str(self.data_root.parent / 'baboon.png')
        dataset[0]['gt_path'] == str(self.data_root.parent / 'baboon.png')

        # test ImgInpaintingDataset
        # test SRFacialLandmarkDataset i.e. SRAnnGTDataset
        dataset = BasicImageDataset(
            ann_file='train.txt',
            metainfo=dict(
                dataset_type='ImgInpaintingDataset', task_name='inpainting'),
            data_root=self.data_root,
            data_prefix=dict(gt='gt'),
            filename_tmpl=dict(),
            pipeline=[])
        assert dataset[0] == dict(
            key='baboon',
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)
        assert dataset[1] == dict(
            key='baboon',
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=1)

    def test_sisr_annotation_dataset(self):
        # setup
        dataset = BasicImageDataset(
            ann_file='train.txt',
            metainfo=dict(
                dataset_type='sisr_annotation_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt'),
            filename_tmpl=dict(img='{}_x4', gt='{}'),
            pipeline=[])

        assert dataset.ann_file == str(self.data_root / 'train.txt')
        assert dataset.data_prefix == dict(
            img=str(self.data_root / 'lq'), gt=str(self.data_root / 'gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='baboon',
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0), dataset[0]

    def test_sisr_folder_dataset(self):
        # setup
        dataset = BasicImageDataset(
            metainfo=dict(
                dataset_type='sisr_folder_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt'),
            filename_tmpl=dict(img='{}_x4', gt='{}'),
            pipeline=[])

        assert dataset.data_prefix == dict(
            img=str(self.data_root / 'lq'), gt=str(self.data_root / 'gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='baboon',
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

    def test_refsr_folder_dataset(self):
        # setup
        dataset = BasicImageDataset(
            metainfo=dict(
                dataset_type='refsr_folder_dataset', task_name='refsr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt', ref='gt'),
            filename_tmpl=dict(img='{}_x4', gt='{}', ref='{}'),
            pipeline=[])

        assert dataset.data_prefix == dict(
            img=str(self.data_root / 'lq'),
            gt=str(self.data_root / 'gt'),
            ref=str(self.data_root / 'gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='baboon',
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            ref_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
