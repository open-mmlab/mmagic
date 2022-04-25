# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from mmcv.utils.testing import assert_dict_has_keys

from mmedit.datasets import (BaseSRDataset, SRAnnotationDataset,
                             SRFacialLandmarkDataset, SRFolderDataset,
                             SRFolderGTDataset, SRFolderMultipleGTDataset,
                             SRFolderRefDataset, SRFolderVideoDataset,
                             SRLmdbDataset, SRREDSDataset,
                             SRREDSMultipleGTDataset, SRTestMultipleGTDataset,
                             SRVid4Dataset, SRVimeo90KDataset,
                             SRVimeo90KMultipleGTDataset)


def mock_open(*args, **kwargs):
    """unittest.mock_open wrapper.

    unittest.mock_open doesn't support iteration. Wrap it to fix this bug.
    Reference: https://stackoverflow.com/a/41656192
    """
    import unittest
    f_open = unittest.mock.mock_open(*args, **kwargs)
    f_open.return_value.__iter__ = lambda self: iter(self.readline, '')
    return f_open


class TestSRDatasets:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = Path(__file__).parent.parent.parent / 'data'

    def test_base_super_resolution_dataset(self):

        class ToyDataset(BaseSRDataset):
            """Toy dataset for testing SRDataset."""

            def __init__(self, pipeline, test_mode=False):
                super().__init__(pipeline, test_mode)

            def load_annotations(self):
                pass

            def __len__(self):
                return 2

        toy_dataset = ToyDataset(pipeline=[])
        file_paths = [
            osp.join('gt', 'baboon.png'),
            osp.join('lq', 'baboon_x4.png')
        ]
        file_paths = [str(self.data_prefix / v) for v in file_paths]

        result = toy_dataset.scan_folder(self.data_prefix)
        assert set(file_paths).issubset(set(result))
        result = toy_dataset.scan_folder(str(self.data_prefix))
        assert set(file_paths).issubset(set(result))

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

        eval_result = toy_dataset.evaluate(results=results)
        assert eval_result == {'PSNR': 25, 'SSIM': 0.7}

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
        assert assert_dict_has_keys(result, target_keys)
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
        assert assert_dict_has_keys(result, target_keys)

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
        assert assert_dict_has_keys(result, target_keys)
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
        assert assert_dict_has_keys(result, target_keys)

    def test_sr_folder_gt_dataset(self):
        # setup
        sr_pipeline = [
            dict(type='LoadImageFromFile', io_backend='disk', key='gt'),
            dict(type='ImageToTensor', keys=['gt'])
        ]
        target_keys = ['gt_path', 'gt']
        gt_folder = self.data_prefix / 'gt'
        filename_tmpl = '{}_x4'

        # input path is Path object
        sr_folder_dataset = SRFolderGTDataset(
            gt_folder=gt_folder,
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl)
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [dict(gt_path=str(gt_folder / 'baboon.png'))]
        result = sr_folder_dataset[0]
        assert (len(sr_folder_dataset) == 1)
        assert assert_dict_has_keys(result, target_keys)
        # input path is str
        sr_folder_dataset = SRFolderGTDataset(
            gt_folder=str(gt_folder),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl)
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [dict(gt_path=str(gt_folder / 'baboon.png'))]
        result = sr_folder_dataset[0]
        assert (len(sr_folder_dataset) == 1)
        assert assert_dict_has_keys(result, target_keys)

    def test_sr_folder_ref_dataset(self):
        # setup
        sr_pipeline = [
            dict(type='LoadImageFromFile', io_backend='disk', key='lq'),
            dict(type='LoadImageFromFile', io_backend='disk', key='gt'),
            dict(type='LoadImageFromFile', io_backend='disk', key='ref'),
            dict(type='PairedRandomCrop', gt_patch_size=128),
            dict(type='ImageToTensor', keys=['lq', 'gt', 'ref'])
        ]
        target_keys = [
            'lq_path', 'gt_path', 'ref_path', 'scale', 'lq', 'gt', 'ref'
        ]
        lq_folder = self.data_prefix / 'lq'
        gt_folder = self.data_prefix / 'gt'
        ref_folder = self.data_prefix / 'gt'
        filename_tmpl = '{}_x4'

        # input path is Path object
        sr_folder_ref_dataset = SRFolderRefDataset(
            lq_folder=lq_folder,
            gt_folder=gt_folder,
            ref_folder=str(ref_folder),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl_lq=filename_tmpl)
        data_infos = sr_folder_ref_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(lq_folder / 'baboon_x4.png'),
                gt_path=str(gt_folder / 'baboon.png'),
                ref_path=str(ref_folder / 'baboon.png'))
        ]
        result = sr_folder_ref_dataset[0]
        assert len(sr_folder_ref_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)
        # input path is str
        sr_folder_ref_dataset = SRFolderRefDataset(
            lq_folder=str(lq_folder),
            gt_folder=str(gt_folder),
            ref_folder=str(ref_folder),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl_lq=filename_tmpl)
        data_infos = sr_folder_ref_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(lq_folder / 'baboon_x4.png'),
                gt_path=str(gt_folder / 'baboon.png'),
                ref_path=str(ref_folder / 'baboon.png'))
        ]
        result = sr_folder_ref_dataset[0]
        assert len(sr_folder_ref_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)

        with pytest.raises(AssertionError):
            sr_folder_ref_dataset = SRFolderRefDataset(
                lq_folder=str(lq_folder),
                gt_folder=str(self.data_prefix / 'image'),  # fake gt_folder
                ref_folder=str(ref_folder),
                pipeline=sr_pipeline,
                scale=4,
                filename_tmpl_lq=filename_tmpl)
        with pytest.raises(AssertionError):
            sr_folder_ref_dataset = SRFolderRefDataset(
                lq_folder=str(self.data_prefix / 'image'),  # fake lq_folder
                gt_folder=str(gt_folder),
                ref_folder=str(ref_folder),
                pipeline=sr_pipeline,
                scale=4,
                filename_tmpl_lq=filename_tmpl)
        with pytest.raises(AssertionError):
            sr_folder_ref_dataset = SRFolderRefDataset(
                lq_folder=str(lq_folder),
                gt_folder=str(self.data_prefix / 'bg'),  # fake gt_folder
                ref_folder=str(ref_folder),
                pipeline=sr_pipeline,
                scale=4,
                filename_tmpl_lq=filename_tmpl)
        with pytest.raises(AssertionError):
            sr_folder_ref_dataset = SRFolderRefDataset(
                lq_folder=str(self.data_prefix / 'bg'),  # fake lq_folder
                gt_folder=str(gt_folder),
                ref_folder=str(ref_folder),
                pipeline=sr_pipeline,
                scale=4,
                filename_tmpl_lq=filename_tmpl)
        with pytest.raises(AssertionError):
            sr_folder_ref_dataset = SRFolderRefDataset(
                lq_folder=None,
                gt_folder=None,
                ref_folder=str(ref_folder),
                pipeline=sr_pipeline,
                scale=4,
                filename_tmpl_lq=filename_tmpl)

    def test_sr_landmark_dataset(self):
        # setup
        sr_pipeline = [
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='color',
                channel_order='rgb',
                backend='cv2')
        ]

        target_keys = ['gt_path', 'bbox', 'shape', 'landmark']
        gt_folder = self.data_prefix / 'face'
        ann_file = self.data_prefix / 'facemark_ann.npy'

        # input path is Path object
        sr_landmark_dataset = SRFacialLandmarkDataset(
            gt_folder=gt_folder,
            ann_file=ann_file,
            pipeline=sr_pipeline,
            scale=4)
        data_infos = sr_landmark_dataset.data_infos
        assert len(data_infos) == 1
        result = sr_landmark_dataset[0]
        assert len(sr_landmark_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)
        # input path is str
        sr_landmark_dataset = SRFacialLandmarkDataset(
            gt_folder=str(gt_folder),
            ann_file=str(ann_file),
            pipeline=sr_pipeline,
            scale=4)
        data_infos = sr_landmark_dataset.data_infos
        assert len(data_infos) == 1
        result = sr_landmark_dataset[0]
        assert len(sr_landmark_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)

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
        assert assert_dict_has_keys(result, target_keys)
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
        assert assert_dict_has_keys(result, target_keys)

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


def test_reds_dataset():
    root_path = Path(__file__).parent.parent.parent / 'data'

    txt_content = ('000/00000001.png (720, 1280, 3)\n'
                   '001/00000001.png (720, 1280, 3)\n'
                   '250/00000001.png (720, 1280, 3)\n')
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
                key=osp.join('000', '00000001'),
                max_frame_num=100,
                num_input_frames=5),
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key=osp.join('001', '00000001'),
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
                key=osp.join('001', '00000001'),
                max_frame_num=100,
                num_input_frames=5),
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key=osp.join('250', '00000001'),
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
                key=osp.join('250', '00000001'),
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
                key=osp.join('000', '00000001'),
                max_frame_num=100,
                num_input_frames=5)
        ]


def test_vimeo90k_dataset():
    root_path = Path(__file__).parent.parent.parent / 'data'

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
            dict(
                lq_path=lq_paths_1,
                gt_path=gt_paths_1,
                key=osp.join('00001', '0266')),
            dict(
                lq_path=lq_paths_2,
                gt_path=gt_paths_2,
                key=osp.join('00002', '0268'))
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
    root_path = Path(__file__).parent.parent.parent / 'data'

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
            metric_average_mode='clip',
            filename_tmpl='{:08d}')

        assert vid4_dataset.data_infos == [
            dict(
                lq_path=str(root_path / 'lq'),
                gt_path=str(root_path / 'gt'),
                key=osp.join('calendar', '00000000'),
                num_input_frames=5,
                max_frame_num=1),
            dict(
                lq_path=str(root_path / 'lq'),
                gt_path=str(root_path / 'gt'),
                key=osp.join('city', '00000000'),
                num_input_frames=5,
                max_frame_num=2),
            dict(
                lq_path=str(root_path / 'lq'),
                gt_path=str(root_path / 'gt'),
                key=osp.join('city', '00000001'),
                num_input_frames=5,
                max_frame_num=2),
        ]

        # test evaluate function ('clip' mode)
        results = [{
            'eval_result': {
                'PSNR': 21,
                'SSIM': 0.75
            }
        }, {
            'eval_result': {
                'PSNR': 22,
                'SSIM': 0.8
            }
        }, {
            'eval_result': {
                'PSNR': 24,
                'SSIM': 0.9
            }
        }]
        eval_result = vid4_dataset.evaluate(results)
        np.testing.assert_almost_equal(eval_result['PSNR'], 22)
        np.testing.assert_almost_equal(eval_result['SSIM'], 0.8)

        # test evaluate function ('all' mode)
        vid4_dataset = SRVid4Dataset(
            lq_folder=root_path / 'lq',
            gt_folder=root_path / 'gt',
            ann_file='fake_ann_file',
            num_input_frames=5,
            pipeline=[],
            scale=4,
            test_mode=False,
            metric_average_mode='all',
            filename_tmpl='{:08d}')
        eval_result = vid4_dataset.evaluate(results)
        np.testing.assert_almost_equal(eval_result['PSNR'], 22.3333333)
        np.testing.assert_almost_equal(eval_result['SSIM'], 0.81666666)

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

        with pytest.raises(ValueError):
            # metric_average_mode can only be either 'folder' or 'all'
            SRVid4Dataset(
                lq_folder=root_path,
                gt_folder=root_path,
                ann_file='fake_ann_file',
                num_input_frames=5,
                pipeline=[],
                scale=4,
                metric_average_mode='abc',
                test_mode=False)

        with pytest.raises(TypeError):
            # results must be a list
            vid4_dataset.evaluate(results=5)
        with pytest.raises(AssertionError):
            # The length of results should be equal to the dataset len
            vid4_dataset.evaluate(results=[results[0]])


def test_sr_reds_multiple_gt_dataset():
    root_path = Path(__file__).parent.parent.parent / 'data'

    # official val partition
    reds_dataset = SRREDSMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=15,
        pipeline=[],
        scale=4,
        val_partition='official',
        test_mode=False)

    assert len(reds_dataset.data_infos) == 240  # 240 training clips
    assert reds_dataset.data_infos[0] == dict(
        lq_path=str(root_path),
        gt_path=str(root_path),
        key='000',
        sequence_length=100,
        num_input_frames=15)

    # REDS4 val partition
    reds_dataset = SRREDSMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=20,
        pipeline=[],
        scale=4,
        val_partition='REDS4',
        test_mode=False)

    assert len(reds_dataset.data_infos) == 266  # 266 training clips
    assert reds_dataset.data_infos[0] == dict(
        lq_path=str(root_path),
        gt_path=str(root_path),
        key='001',
        sequence_length=100,
        num_input_frames=20)  # 000 is been removed

    with pytest.raises(ValueError):
        # wrong val_partitaion
        reds_dataset = SRREDSMultipleGTDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            num_input_frames=5,
            pipeline=[],
            scale=4,
            val_partition='wrong_val_partition',
            test_mode=False)

    # test mode
    # official val partition
    reds_dataset = SRREDSMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=5,
        pipeline=[],
        scale=4,
        val_partition='official',
        test_mode=True)

    assert len(reds_dataset.data_infos) == 30  # 30 test clips
    assert reds_dataset.data_infos[0] == dict(
        lq_path=str(root_path),
        gt_path=str(root_path),
        key='240',
        sequence_length=100,
        num_input_frames=5)

    # REDS4 val partition
    reds_dataset = SRREDSMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=5,
        pipeline=[],
        scale=4,
        val_partition='REDS4',
        test_mode=True)

    assert len(reds_dataset.data_infos) == 4  # 4 test clips
    assert reds_dataset.data_infos[1] == dict(
        lq_path=str(root_path),
        gt_path=str(root_path),
        key='011',
        sequence_length=100,
        num_input_frames=5)

    # REDS4 val partition (repeat > 1)
    reds_dataset = SRREDSMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=5,
        pipeline=[],
        scale=4,
        val_partition='REDS4',
        repeat=2,
        test_mode=True)

    assert len(reds_dataset.data_infos) == 8  # 4 test clips
    assert reds_dataset.data_infos[5] == dict(
        lq_path=str(root_path),
        gt_path=str(root_path),
        key='011',
        sequence_length=100,
        num_input_frames=5)

    # REDS4 val partition (repeat != int)
    with pytest.raises(TypeError):
        SRREDSMultipleGTDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            num_input_frames=5,
            pipeline=[],
            scale=4,
            val_partition='REDS4',
            repeat=1.5,
            test_mode=True)


def test_sr_vimeo90k_mutiple_gt_dataset():
    root_path = Path(__file__).parent.parent.parent / 'data' / 'vimeo90k'

    txt_content = ('00001/0266 (256,448,3)\n')
    mocked_open_function = mock_open(read_data=txt_content)

    num_input_frames = 5
    lq_paths = [
        str(root_path / '00001' / '0266' / f'im{v}.png')
        for v in range(1, num_input_frames + 1)
    ]
    gt_paths = [
        str(root_path / '00001' / '0266' / f'im{v}.png')
        for v in range(1, num_input_frames + 1)
    ]

    with patch('builtins.open', mocked_open_function):
        vimeo90k_dataset = SRVimeo90KMultipleGTDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            ann_file='fake_ann_file',
            pipeline=[],
            scale=4,
            num_input_frames=num_input_frames,
            test_mode=False)
        assert vimeo90k_dataset.data_infos == [
            dict(
                lq_path=lq_paths,
                gt_path=gt_paths,
                key=osp.join('00001', '0266'))
        ]


def test_sr_test_multiple_gt_dataset():
    root_path = Path(
        __file__).parent.parent.parent / 'data' / 'test_multiple_gt'

    test_dataset = SRTestMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        pipeline=[],
        scale=4,
        test_mode=True)

    assert test_dataset.data_infos == [
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key='sequence_1',
            sequence_length=2),
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key='sequence_2',
            sequence_length=1)
    ]


def test_sr_folder_multiple_gt_dataset():
    root_path = Path(
        __file__).parent.parent.parent / 'data' / 'test_multiple_gt'

    # test without num_input_frames
    test_dataset = SRFolderMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        pipeline=[],
        scale=4,
        test_mode=True)
    assert test_dataset.data_infos == [
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key='sequence_1',
            num_input_frames=2,
            sequence_length=2),
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key='sequence_2',
            num_input_frames=1,
            sequence_length=1)
    ]

    # test with num_input_frames
    test_dataset = SRFolderMultipleGTDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        pipeline=[],
        scale=4,
        num_input_frames=1,
        test_mode=True)
    assert test_dataset.data_infos == [
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key='sequence_1',
            num_input_frames=1,
            sequence_length=2),
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key='sequence_2',
            num_input_frames=1,
            sequence_length=1)
    ]

    # with annotation file (without num_input_frames)
    txt_content = ('sequence_1 2\n')
    mocked_open_function = mock_open(read_data=txt_content)
    with patch('builtins.open', mocked_open_function):
        # test without num_input_frames
        test_dataset = SRFolderMultipleGTDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            pipeline=[],
            scale=4,
            ann_file='fake_ann_file',
            test_mode=True)
        assert test_dataset.data_infos == [
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='sequence_1',
                num_input_frames=2,
                sequence_length=2),
        ]

        # with annotation file (with num_input_frames)
        test_dataset = SRFolderMultipleGTDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            pipeline=[],
            scale=4,
            ann_file='fake_ann_file',
            num_input_frames=1,
            test_mode=True)
        assert test_dataset.data_infos == [
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key='sequence_1',
                num_input_frames=1,
                sequence_length=2),
        ]

    # num_input_frames must be a positive integer
    with pytest.raises(ValueError):
        SRFolderMultipleGTDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            pipeline=[],
            scale=4,
            num_input_frames=-1,
            test_mode=True)


def test_sr_folder_video_dataset():
    root_path = Path(
        __file__).parent.parent.parent / 'data' / 'test_multiple_gt'

    test_dataset = SRFolderVideoDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=5,
        pipeline=[],
        scale=4,
        test_mode=True)

    assert test_dataset.data_infos == [
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key=osp.join('sequence_1', '00000000'),
            num_input_frames=5,
            max_frame_num=2),
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key=osp.join('sequence_1', '00000001'),
            num_input_frames=5,
            max_frame_num=2),
        dict(
            lq_path=str(root_path),
            gt_path=str(root_path),
            key=osp.join('sequence_2', '00000000'),
            num_input_frames=5,
            max_frame_num=1),
    ]

    # with annotation file
    txt_content = ('sequence_1/00000000 2\n')
    mocked_open_function = mock_open(read_data=txt_content)
    with patch('builtins.open', mocked_open_function):
        test_dataset = SRFolderVideoDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            num_input_frames=5,
            pipeline=[],
            scale=4,
            ann_file='fake_ann_file',
            test_mode=True)
        assert test_dataset.data_infos == [
            dict(
                lq_path=str(root_path),
                gt_path=str(root_path),
                key=osp.join('sequence_1', '00000000'),
                num_input_frames=5,
                max_frame_num=2),
        ]

    # test evaluate function ('clip' mode)
    test_dataset = SRFolderVideoDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=5,
        pipeline=[],
        scale=4,
        metric_average_mode='clip',
        test_mode=True)
    results = [{
        'eval_result': {
            'PSNR': 21,
            'SSIM': 0.75
        }
    }, {
        'eval_result': {
            'PSNR': 23,
            'SSIM': 0.85
        }
    }, {
        'eval_result': {
            'PSNR': 24,
            'SSIM': 0.9
        }
    }]
    eval_result = test_dataset.evaluate(results)
    np.testing.assert_almost_equal(eval_result['PSNR'], 23)
    np.testing.assert_almost_equal(eval_result['SSIM'], 0.85)

    # test evaluate function ('all' mode)
    test_dataset = SRFolderVideoDataset(
        lq_folder=root_path,
        gt_folder=root_path,
        num_input_frames=5,
        pipeline=[],
        scale=4,
        metric_average_mode='all',
        test_mode=True)
    eval_result = test_dataset.evaluate(results)
    np.testing.assert_almost_equal(eval_result['PSNR'], 22.6666666)
    np.testing.assert_almost_equal(eval_result['SSIM'], 0.83333333)

    # num_input_frames should be odd numbers
    with pytest.raises(AssertionError):
        SRFolderVideoDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            num_input_frames=6,
            pipeline=[],
            scale=4,
            test_mode=True)

    # metric_average_mode can only be either 'folder' or 'all'
    with pytest.raises(ValueError):
        SRFolderVideoDataset(
            lq_folder=root_path,
            gt_folder=root_path,
            num_input_frames=5,
            pipeline=[],
            scale=4,
            metric_average_mode='abc',
            test_mode=False)

    # results must be a list
    with pytest.raises(TypeError):
        test_dataset.evaluate(results=5)

    # The length of results should be equal to the dataset len
    with pytest.raises(AssertionError):
        test_dataset.evaluate(results=[results[0]])
