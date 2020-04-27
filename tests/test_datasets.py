import os.path as osp
from pathlib import Path

import numpy as np
import pytest
from mmedit.datasets import (AdobeComp1kDataset, BaseSRDataset, RepeatDataset,
                             SRAnnotationDataset, SRFolderDataset,
                             SRLmdbDataset)
from torch.utils.data import Dataset


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


class TestMattingDatasets(object):

    @classmethod
    def setup_class(cls):
        # creat para for creating a dataset.
        cls.data_prefix = Path(__file__).parent / 'data'
        cls.ann_file = osp.join(cls.data_prefix, 'test_list.json')
        cls.pipeline = [dict(type='LoadAlpha', key='alpha', flag='grayscale')]

    def test_comp1k_dataset(self):
        comp1k_dataset = AdobeComp1kDataset(self.ann_file, self.pipeline,
                                            self.data_prefix)
        first_data = comp1k_dataset[0]

        assert 'alpha' in first_data
        assert isinstance(first_data['alpha'], np.ndarray)
        assert first_data['alpha'].shape == (552, 800)

    def test_comp1k_evaluate(self):
        comp1k_dataset = AdobeComp1kDataset(self.ann_file, self.pipeline,
                                            self.data_prefix)

        with pytest.raises(TypeError):
            comp1k_dataset.evaluate('Not a list object')

        results = [(None, {
            'SAD': 26,
            'MSE': 0.006
        }), (None, {
            'SAD': 24,
            'MSE': 0.004
        })]
        eval_result = comp1k_dataset.evaluate(results)
        assert set(eval_result.keys()) == set(['SAD', 'MSE'])
        assert eval_result['SAD'] == 25
        assert eval_result['MSE'] == 0.005


class TestSRDatasets(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = Path(__file__).parent / 'data'

    def test_base_super_resolution_dataset(self):

        class ToyDataset(BaseSRDataset):
            """Toy dataset for testing SRDataset."""

            def __init__(self, pipeline, test_mode=False):
                super(ToyDataset, self).__init__(pipeline, test_mode)

            def load_annotations(self):
                pass

            def __len__(self):
                return 2

        toy_dataset = ToyDataset(pipeline=[])
        file_paths = ['gt/baboon.png', 'lq/baboon_x4.png']
        file_paths = [str(self.data_prefix / v) for v in file_paths]

        result = toy_dataset.scan_folder(self.data_prefix)
        assert check_keys_contain(result, file_paths)
        result = toy_dataset.scan_folder(str(self.data_prefix))
        assert check_keys_contain(result, file_paths)

        with pytest.raises(TypeError):
            toy_dataset.scan_folder(123)

        # test evaluate function
        results = [{
            'eval_result': {
                'PSNR': 20,
                'SSIM': 0.6
            }
        }, {
            'eval_result': {
                'PSNR': 30,
                'SSIM': 0.8
            }
        }]

        with pytest.raises(TypeError):
            # results must be a list
            toy_dataset.evaluate(results=5)
        with pytest.raises(AssertionError):
            # The length of results should be equal to the dataset len
            toy_dataset.evaluate(results=[results[0]])

        eval_results = toy_dataset.evaluate(results=results)
        assert eval_results == {'PSNR': 25, 'SSIM': 0.7}

        with pytest.raises(AssertionError):
            results = [{
                'eval_result': {
                    'PSNR': 20,
                    'SSIM': 0.6
                }
            }, {
                'eval_result': {
                    'PSNR': 30
                }
            }]
            # Length of evaluation result should be the same as the dataset len
            toy_dataset.evaluate(results=results)

    def test_sr_annotation_dataset(self):
        # setup
        anno_file_path = self.data_prefix / 'train.txt'
        sr_pipeline = [
            dict(type='LoadImageFromFile', io_backend='disk', key='lq'),
            dict(type='LoadImageFromFile', io_backend='disk', key='gt'),
            dict(type='PairedRandomCrop', gt_patch_size=128),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ]
        target_keys = [
            'lq_path', 'gt_path', 'scale', 'lq', 'lq_ori_shape', 'gt',
            'gt_ori_shape'
        ]

        # input path is Path object
        sr_annotation_dataset = SRAnnotationDataset(
            ann_file=anno_file_path,
            pipeline=sr_pipeline,
            scale=4,
            data_prefix=self.data_prefix)
        data_infos = sr_annotation_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(self.data_prefix / 'lq' / 'baboon_x4.png'),
                gt_path=str(self.data_prefix / 'gt' / 'baboon.png'))
        ]
        result = sr_annotation_dataset[0]
        assert (len(sr_annotation_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)
        # input path is str
        sr_annotation_dataset = SRAnnotationDataset(
            ann_file=str(anno_file_path),
            pipeline=sr_pipeline,
            scale=4,
            data_prefix=str(self.data_prefix))
        data_infos = sr_annotation_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(self.data_prefix / 'lq' / 'baboon_x4.png'),
                gt_path=str(self.data_prefix / 'gt' / 'baboon.png'))
        ]
        result = sr_annotation_dataset[0]
        assert (len(sr_annotation_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)

    def test_sr_folder_dataset(self):
        # setup
        sr_pipeline = [
            dict(type='LoadImageFromFile', io_backend='disk', key='lq'),
            dict(type='LoadImageFromFile', io_backend='disk', key='gt'),
            dict(type='PairedRandomCrop', gt_patch_size=128),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ]
        target_keys = ['lq_path', 'gt_path', 'scale', 'lq', 'gt']
        lq_folder = self.data_prefix / 'lq'
        gt_folder = self.data_prefix / 'gt'
        filename_tmpl = '{}_x4'

        # input path is Path object
        sr_folder_dataset = SRFolderDataset(
            lq_folder=lq_folder,
            gt_folder=gt_folder,
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl)
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(lq_folder / 'baboon_x4.png'),
                gt_path=str(gt_folder / 'baboon.png'))
        ]
        result = sr_folder_dataset[0]
        assert (len(sr_folder_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)
        # input path is str
        sr_folder_dataset = SRFolderDataset(
            lq_folder=str(lq_folder),
            gt_folder=str(gt_folder),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl)
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(lq_folder / 'baboon_x4.png'),
                gt_path=str(gt_folder / 'baboon.png'))
        ]
        result = sr_folder_dataset[0]
        assert (len(sr_folder_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)

    def test_sr_lmdb_dataset(self):
        # setup
        lq_lmdb_folder = self.data_prefix / 'lq.lmdb'
        sr_pipeline = [
            dict(
                type='LoadImageFromFile',
                io_backend='lmdb',
                key='lq',
                db_path=lq_lmdb_folder),
            dict(
                type='LoadImageFromFile',
                io_backend='lmdb',
                key='gt',
                db_path=lq_lmdb_folder),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ]
        target_keys = [
            'lq_path', 'gt_path', 'scale', 'lq', 'lq_ori_shape', 'gt',
            'gt_ori_shape'
        ]

        # input path is Path object
        sr_lmdb_dataset = SRLmdbDataset(
            lq_folder=lq_lmdb_folder,
            gt_folder=lq_lmdb_folder,  # fake gt_folder
            pipeline=sr_pipeline,
            scale=1)
        data_infos = sr_lmdb_dataset.data_infos
        assert data_infos == [dict(lq_path='baboon', gt_path='baboon')]
        result = sr_lmdb_dataset[0]
        assert (len(sr_lmdb_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)
        # input path is str
        sr_lmdb_dataset = SRLmdbDataset(
            lq_folder=str(lq_lmdb_folder),
            gt_folder=(lq_lmdb_folder),  # fake gt_folder
            pipeline=sr_pipeline,
            scale=1)
        data_infos = sr_lmdb_dataset.data_infos
        assert data_infos == [dict(lq_path='baboon', gt_path='baboon')]
        result = sr_lmdb_dataset[0]
        assert (len(sr_lmdb_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)

        with pytest.raises(ValueError):
            sr_lmdb_dataset = SRLmdbDataset(
                lq_folder=self.data_prefix,  # normal folder
                gt_folder=lq_lmdb_folder,  # fake gt_folder
                pipeline=sr_pipeline,
                scale=1)
        with pytest.raises(ValueError):
            sr_lmdb_dataset = SRLmdbDataset(
                lq_folder=str(self.data_prefix),  # normal folder
                gt_folder=lq_lmdb_folder,  # fake gt_folder
                pipeline=sr_pipeline,
                scale=1)
        with pytest.raises(ValueError):
            sr_lmdb_dataset = SRLmdbDataset(
                lq_folder=lq_lmdb_folder,
                gt_folder=self.data_prefix,  # normal folder
                pipeline=sr_pipeline,
                scale=1)
        with pytest.raises(ValueError):
            sr_lmdb_dataset = SRLmdbDataset(
                lq_folder=lq_lmdb_folder,
                gt_folder=str(self.data_prefix),  # normal folder
                pipeline=sr_pipeline,
                scale=1)


def test_repeat_dataset():

    class ToyDataset(Dataset):

        def __init__(self):
            super(ToyDataset, self).__init__()
            self.members = [1, 2, 3, 4, 5]

        def __len__(self):
            return len(self.members)

        def __getitem__(self, idx):
            return self.members[idx % 5]

    toy_dataset = ToyDataset()
    repeat_dataset = RepeatDataset(toy_dataset, 2)
    assert len(repeat_dataset) == 10
    assert repeat_dataset[2] == 3
    assert repeat_dataset[8] == 4
