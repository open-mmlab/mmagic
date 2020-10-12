import os.path as osp
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from torch.utils.data import Dataset

from mmedit.datasets import (AdobeComp1kDataset, BaseGenerationDataset,
                             BaseSRDataset, GenerationPairedDataset,
                             GenerationUnpairedDataset, RepeatDataset,
                             SRAnnotationDataset, SRFolderDataset,
                             SRLmdbDataset, SRREDSDataset, SRVid4Dataset,
                             SRVimeo90KDataset)


def mock_open(*args, **kargs):
    """unittest.mock_open wrapper.

    unittest.mock_open doesn't support iteration. Wrap it to fix this bug.
    Reference: https://stackoverflow.com/a/41656192
    """
    import unittest
    f_open = unittest.mock.mock_open(*args, **kargs)
    f_open.return_value.__iter__ = lambda self: iter(self.readline, '')
    return f_open


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


class TestMattingDatasets(object):

    @classmethod
    def setup_class(cls):
        # creat para for creating a dataset.
        cls.data_prefix = Path(__file__).parent / 'data'
        cls.ann_file = osp.join(cls.data_prefix, 'test_list.json')
        cls.pipeline = [
            dict(type='LoadImageFromFile', key='alpha', flag='grayscale')
        ]

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

        results = [{
            'pred_alpha': None,
            'eval_result': {
                'SAD': 26,
                'MSE': 0.006
            }
        }, {
            'pred_alpha': None,
            'eval_result': {
                'SAD': 24,
                'MSE': 0.004
            }
        }]

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
            lq_folder=self.data_prefix / 'lq',
            gt_folder=self.data_prefix / 'gt',
            ann_file=anno_file_path,
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl='{}_x4')
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
            lq_folder=str(self.data_prefix / 'lq'),
            gt_folder=str(self.data_prefix / 'gt'),
            ann_file=str(anno_file_path),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl='{}_x4')
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


class TestGenerationDatasets(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = Path(__file__).parent / 'data'

    def test_base_generation_dataset(self):

        class ToyDataset(BaseGenerationDataset):
            """Toy dataset for testing Generation Dataset."""

            def load_annotations(self):
                pass

        toy_dataset = ToyDataset(pipeline=[])
        file_paths = [
            'paired/test/3.jpg', 'paired/train/1.jpg', 'paired/train/2.jpg'
        ]
        file_paths = [str(self.data_prefix / v) for v in file_paths]

        # test scan_folder
        result = toy_dataset.scan_folder(self.data_prefix)
        assert check_keys_contain(result, file_paths)
        result = toy_dataset.scan_folder(str(self.data_prefix))
        assert check_keys_contain(result, file_paths)

        with pytest.raises(TypeError):
            toy_dataset.scan_folder(123)

        # test evaluate
        toy_dataset.data_infos = file_paths
        with pytest.raises(TypeError):
            _ = toy_dataset.evaluate(1)
        test_results = [dict(saved_flag=True), dict(saved_flag=True)]
        with pytest.raises(AssertionError):
            _ = toy_dataset.evaluate(test_results)
        test_results = [
            dict(saved_flag=True),
            dict(saved_flag=True),
            dict(saved_flag=False)
        ]
        eval_results = toy_dataset.evaluate(test_results)
        assert eval_results['val_saved_number'] == 2

    def test_generation_paired_dataset(self):
        # setup
        img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        pipeline = [
            dict(
                type='LoadPairedImageFromFile',
                io_backend='disk',
                key='pair',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_a', 'img_b'],
                scale=(286, 286),
                interpolation='bicubic'),
            dict(
                type='FixedCrop',
                keys=['img_a', 'img_b'],
                crop_size=(256, 256)),
            dict(type='Flip', keys=['img_a', 'img_b'], direction='horizontal'),
            dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
            dict(
                type='Normalize',
                keys=['img_a', 'img_b'],
                to_rgb=True,
                **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img_a', 'img_b']),
            dict(
                type='Collect',
                keys=['img_a', 'img_b'],
                meta_keys=['img_a_path', 'img_b_path'])
        ]
        target_keys = ['img_a', 'img_b', 'meta']
        target_meta_keys = ['img_a_path', 'img_b_path']
        pair_folder = self.data_prefix / 'paired'

        # input path is Path object
        generation_paried_dataset = GenerationPairedDataset(
            dataroot=pair_folder, pipeline=pipeline, test_mode=True)
        data_infos = generation_paried_dataset.data_infos
        assert data_infos == [
            dict(pair_path=str(pair_folder / 'test' / '3.jpg'))
        ]
        result = generation_paried_dataset[0]
        assert (len(generation_paried_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(pair_folder / 'test' /
                                                         '3.jpg'))
        assert (result['meta'].data['img_b_path'] == str(pair_folder / 'test' /
                                                         '3.jpg'))

        # input path is str
        generation_paried_dataset = GenerationPairedDataset(
            dataroot=str(pair_folder), pipeline=pipeline, test_mode=True)
        data_infos = generation_paried_dataset.data_infos
        assert data_infos == [
            dict(pair_path=str(pair_folder / 'test' / '3.jpg'))
        ]
        result = generation_paried_dataset[0]
        assert (len(generation_paried_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(pair_folder / 'test' /
                                                         '3.jpg'))
        assert (result['meta'].data['img_b_path'] == str(pair_folder / 'test' /
                                                         '3.jpg'))

        # test_mode = False
        generation_paried_dataset = GenerationPairedDataset(
            dataroot=str(pair_folder), pipeline=pipeline, test_mode=False)
        data_infos = generation_paried_dataset.data_infos
        assert data_infos == [
            dict(pair_path=str(pair_folder / 'train' / '1.jpg')),
            dict(pair_path=str(pair_folder / 'train' / '2.jpg'))
        ]
        assert (len(generation_paried_dataset) == 2)
        result = generation_paried_dataset[0]
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(pair_folder /
                                                         'train' / '1.jpg'))
        assert (result['meta'].data['img_b_path'] == str(pair_folder /
                                                         'train' / '1.jpg'))
        result = generation_paried_dataset[1]
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(pair_folder /
                                                         'train' / '2.jpg'))
        assert (result['meta'].data['img_b_path'] == str(pair_folder /
                                                         'train' / '2.jpg'))

    def test_generation_unpaired_dataset(self):
        # setup
        img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        pipeline = [
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_a',
                flag='color'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_b',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_a', 'img_b'],
                scale=(286, 286),
                interpolation='bicubic'),
            dict(
                type='Crop',
                keys=['img_a', 'img_b'],
                crop_size=(256, 256),
                random_crop=True),
            dict(type='Flip', keys=['img_a'], direction='horizontal'),
            dict(type='Flip', keys=['img_b'], direction='horizontal'),
            dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
            dict(
                type='Normalize',
                keys=['img_a', 'img_b'],
                to_rgb=True,
                **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img_a', 'img_b']),
            dict(
                type='Collect',
                keys=['img_a', 'img_b'],
                meta_keys=['img_a_path', 'img_b_path'])
        ]
        target_keys = ['img_a', 'img_b', 'meta']
        target_meta_keys = ['img_a_path', 'img_b_path']
        unpair_folder = self.data_prefix / 'unpaired'

        # input path is Path object
        generation_unparied_dataset = GenerationUnpairedDataset(
            dataroot=unpair_folder, pipeline=pipeline, test_mode=True)
        data_infos_a = generation_unparied_dataset.data_infos_a
        data_infos_b = generation_unparied_dataset.data_infos_b
        assert data_infos_a == [
            dict(path=str(unpair_folder / 'testA' / '5.jpg'))
        ]
        assert data_infos_b == [
            dict(path=str(unpair_folder / 'testB' / '6.jpg'))
        ]
        result = generation_unparied_dataset[0]
        assert (len(generation_unparied_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'testA' / '5.jpg'))
        assert (result['meta'].data['img_b_path'] == str(unpair_folder /
                                                         'testB' / '6.jpg'))

        # input path is str
        generation_unparied_dataset = GenerationUnpairedDataset(
            dataroot=str(unpair_folder), pipeline=pipeline, test_mode=True)
        data_infos_a = generation_unparied_dataset.data_infos_a
        data_infos_b = generation_unparied_dataset.data_infos_b
        assert data_infos_a == [
            dict(path=str(unpair_folder / 'testA' / '5.jpg'))
        ]
        assert data_infos_b == [
            dict(path=str(unpair_folder / 'testB' / '6.jpg'))
        ]
        result = generation_unparied_dataset[0]
        assert (len(generation_unparied_dataset) == 1)
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'testA' / '5.jpg'))
        assert (result['meta'].data['img_b_path'] == str(unpair_folder /
                                                         'testB' / '6.jpg'))

        # test_mode = False
        generation_unparied_dataset = GenerationUnpairedDataset(
            dataroot=str(unpair_folder), pipeline=pipeline, test_mode=False)
        data_infos_a = generation_unparied_dataset.data_infos_a
        data_infos_b = generation_unparied_dataset.data_infos_b
        assert data_infos_a == [
            dict(path=str(unpair_folder / 'trainA' / '1.jpg')),
            dict(path=str(unpair_folder / 'trainA' / '2.jpg'))
        ]
        assert data_infos_b == [
            dict(path=str(unpair_folder / 'trainB' / '3.jpg')),
            dict(path=str(unpair_folder / 'trainB' / '4.jpg'))
        ]
        assert (len(generation_unparied_dataset) == 2)
        img_b_paths = [
            str(unpair_folder / 'trainB' / '3.jpg'),
            str(unpair_folder / 'trainB' / '4.jpg')
        ]
        result = generation_unparied_dataset[0]
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'trainA' / '1.jpg'))
        assert result['meta'].data['img_b_path'] in img_b_paths
        result = generation_unparied_dataset[1]
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['meta'].data.keys(), target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'trainA' / '2.jpg'))
        assert result['meta'].data['img_b_path'] in img_b_paths


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


def test_reds_dataset():
    root_path = Path(__file__).parent / 'data'

    txt_content = ('000/00000001 (720, 1280, 3)\n'
                   '001/00000001 (720, 1280, 3)\n'
                   '250/00000001 (720, 1280, 3)\n')
    mocked_open_function = mock_open(read_data=txt_content)

    with patch('builtins.open', mocked_open_function):
        # official val partition
        reds_dataset = SRREDSDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            ann_file='fake_ann_file',
            num_input_frames=5,
            pipeline=[],
            scale=4,
            val_partition='official',
            test_mode=False)

        assert reds_dataset.data_infos == [
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='000/00000001',
                max_frame_num=100,
                num_input_frames=5),
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='001/00000001',
                max_frame_num=100,
                num_input_frames=5)
        ]

        # REDS4 val partition
        reds_dataset = SRREDSDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            ann_file='fake_ann_file',
            num_input_frames=5,
            pipeline=[],
            scale=4,
            val_partition='REDS4',
            test_mode=False)

        assert reds_dataset.data_infos == [
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='001/00000001',
                max_frame_num=100,
                num_input_frames=5),
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='250/00000001',
                max_frame_num=100,
                num_input_frames=5)
        ]

        with pytest.raises(ValueError):
            # wrong val_partitaion
            reds_dataset = SRREDSDataset(
                lq_folder=root_path,
                gt_folder=root_path,
                ann_file='fake_ann_file',
                num_input_frames=5,
                pipeline=[],
                scale=4,
                val_partition='wrong_val_partition',
                test_mode=False)

        with pytest.raises(AssertionError):
            # num_input_frames should be odd numbers
            reds_dataset = SRREDSDataset(
                lq_folder=root_path,
                gt_folder=root_path,
                ann_file='fake_ann_file',
                num_input_frames=6,
                pipeline=[],
                scale=4,
                val_partition='wrong_val_partition',
                test_mode=False)

        # test mode
        # official val partition
        reds_dataset = SRREDSDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            ann_file='fake_ann_file',
            num_input_frames=5,
            pipeline=[],
            scale=4,
            val_partition='official',
            test_mode=True)

        assert reds_dataset.data_infos == [
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='250/00000001',
                max_frame_num=100,
                num_input_frames=5)
        ]
        # REDS4 val partition
        reds_dataset = SRREDSDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            ann_file='fake_ann_file',
            num_input_frames=5,
            pipeline=[],
            scale=4,
            val_partition='REDS4',
            test_mode=True)

        assert reds_dataset.data_infos == [
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='000/00000001',
                max_frame_num=100,
                num_input_frames=5)
        ]


def test_vimeo90k_dataset():
    root_path = Path(__file__).parent / 'data'

    txt_content = ('00001/0266 (256, 448, 3)\n00002/0268 (256, 448, 3)\n')
    mocked_open_function = mock_open(read_data=txt_content)
    lq_paths_1 = [
        str(root_path / '00001' / '0266' / f'im{v}.png') for v in range(1, 8)
    ]
    gt_paths_1 = [str(root_path / '00001' / '0266' / 'im4.png')]
    lq_paths_2 = [
        str(root_path / '00002' / '0268' / f'im{v}.png') for v in range(1, 8)
    ]
    gt_paths_2 = [str(root_path / '00002' / '0268' / 'im4.png')]
    with patch('builtins.open', mocked_open_function):
        vimeo90k_dataset = SRVimeo90KDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            ann_file='fake_ann_file',
            num_input_frames=7,
            pipeline=[],
            scale=4,
            test_mode=False)

        assert vimeo90k_dataset.data_infos == [
            dict(lq_path=lq_paths_1, gt_path=gt_paths_1, key='00001/0266'),
            dict(lq_path=lq_paths_2, gt_path=gt_paths_2, key='00002/0268')
        ]

        with pytest.raises(AssertionError):
            # num_input_frames should be odd numbers
            vimeo90k_dataset = SRVimeo90KDataset(
                lq_folder=root_path,
                gt_folder=root_path,
                ann_file='fake_ann_file',
                num_input_frames=6,
                pipeline=[],
                scale=4,
                test_mode=False)


def test_vid4_dataset():
    root_path = Path(__file__).parent / 'data'

    txt_content = ('calendar 1 (320,480,3)\ncity 2 (320,480,3)\n')
    mocked_open_function = mock_open(read_data=txt_content)

    with patch('builtins.open', mocked_open_function):
        vid4_dataset = SRVid4Dataset(
            lq_folder=root_path / 'lq',
            gt_folder=root_path / 'gt',
            ann_file='fake_ann_file',
            num_input_frames=5,
            pipeline=[],
            scale=4,
            test_mode=False,
            filename_tmpl='{:08d}')

        assert vid4_dataset.data_infos == [
            dict(
                lq_path=str(root_path / 'lq'),
                gt_path=str(root_path / 'gt'),
                key='calendar/00000000',
                num_input_frames=5,
                max_frame_num=1),
            dict(
                lq_path=str(root_path / 'lq'),
                gt_path=str(root_path / 'gt'),
                key='city/00000000',
                num_input_frames=5,
                max_frame_num=2),
            dict(
                lq_path=str(root_path / 'lq'),
                gt_path=str(root_path / 'gt'),
                key='city/00000001',
                num_input_frames=5,
                max_frame_num=2),
        ]

        with pytest.raises(AssertionError):
            # num_input_frames should be odd numbers
            SRVid4Dataset(
                lq_folder=root_path,
                gt_folder=root_path,
                ann_file='fake_ann_file',
                num_input_frames=6,
                pipeline=[],
                scale=4,
                test_mode=False)
