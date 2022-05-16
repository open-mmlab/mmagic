# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import pytest

from mmedit.datasets import BaseImageDataset


class TestImageDatasets:

    @classmethod
    def setup_class(cls):
        cls.data_root = Path(__file__).parent.parent / 'data' / 'image'

    def test_version_1_method(self):

        # test SRAnnotationDataset
        dataset = BaseImageDataset(
            ann_file=self.data_root / 'train.txt',
            metainfo=dict(
                dataset_type='sr_annotation_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt'),
            filename_tmpl=dict(img='{}_x4'),
            pipeline=[])
        assert dataset[0] == dict(
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

        # test SRFolderDataset
        dataset = BaseImageDataset(
            ann_file='',
            metainfo=dict(dataset_type='sr_folder_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt'),
            filename_tmpl=dict(img='{}_x4'),
            pipeline=[])
        assert dataset[0] == dict(
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

        # test SRFolderGTDataset
        dataset = BaseImageDataset(
            ann_file='',
            metainfo=dict(dataset_type='sr_folder_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(gt='gt'),
            pipeline=[])
        assert dataset[0] == dict(
            gt_path=str(self.data_root / 'gt' / 'baboon.png'), sample_idx=0)

        # test SRLmdbDataset
        # TODO wait for a solution.
        # modify loading or data_list in dataset.

        # test ImgInpaintingDataset
        # test SRFacialLandmarkDataset i.e. SRAnnGTDataset
        dataset = BaseImageDataset(
            ann_file='train.txt',
            metainfo=dict(
                dataset_type='ImgInpaintingDataset', task_name='inpainting'),
            data_root=self.data_root,
            data_prefix=dict(gt='gt'),
            pipeline=[])
        assert dataset[0] == dict(
            gt_path=str(self.data_root / 'gt' / 'baboon.png'), sample_idx=0)

    def test_sisr_annotation_dataset(self):
        # setup
        dataset = BaseImageDataset(
            ann_file=self.data_root / 'train.txt',
            metainfo=dict(
                dataset_type='sisr_annotation_dataset', task_name='sisr'),
            data_root=self.data_root,
            data_prefix=dict(img='lq', gt='gt'),
            filename_tmpl=dict(img='{}_x4', gt='{}'),
            pipeline=[])

        assert dataset.ann_file == self.data_root / 'train.txt'
        assert dataset.data_prefix == dict(
            img=str(self.data_root / 'lq'), gt=str(self.data_root / 'gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

    def test_sisr_folder_dataset(self):
        # setup
        dataset = BaseImageDataset(
            ann_file='',
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
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

    def test_refsr_folder_dataset(self):
        # setup
        dataset = BaseImageDataset(
            ann_file='',
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
            img_path=str(self.data_root / 'lq' / 'baboon_x4.png'),
            gt_path=str(self.data_root / 'gt' / 'baboon.png'),
            ref_path=str(self.data_root / 'gt' / 'baboon.png'),
            sample_idx=0)

    def test_assert(self):

        with pytest.raises(AssertionError):
            BaseImageDataset(
                data_prefix=dict(img='', gt=''),
                filename_tmpl=dict(img='{}', ggt='{}'))
