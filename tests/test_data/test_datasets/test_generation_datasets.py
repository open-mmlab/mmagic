# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import pytest
from mmcv.utils.testing import assert_dict_has_keys

from mmedit.datasets import (BaseGenerationDataset, GenerationPairedDataset,
                             GenerationUnpairedDataset)


class TestGenerationDatasets:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = Path(__file__).parent.parent.parent / 'data'

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
        assert set(file_paths).issubset(set(result))
        result = toy_dataset.scan_folder(str(self.data_prefix))
        assert set(file_paths).issubset(set(result))

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
        eval_result = toy_dataset.evaluate(test_results)
        assert eval_result['val_saved_number'] == 2

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
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
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
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
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
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(pair_folder /
                                                         'train' / '1.jpg'))
        assert (result['meta'].data['img_b_path'] == str(pair_folder /
                                                         'train' / '1.jpg'))
        result = generation_paried_dataset[1]
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
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
        generation_unpaired_dataset = GenerationUnpairedDataset(
            dataroot=unpair_folder, pipeline=pipeline, test_mode=True)
        data_infos_a = generation_unpaired_dataset.data_infos_a
        data_infos_b = generation_unpaired_dataset.data_infos_b
        assert data_infos_a == [
            dict(path=str(unpair_folder / 'testA' / '5.jpg'))
        ]
        assert data_infos_b == [
            dict(path=str(unpair_folder / 'testB' / '6.jpg'))
        ]
        result = generation_unpaired_dataset[0]
        assert (len(generation_unpaired_dataset) == 1)
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'testA' / '5.jpg'))
        assert (result['meta'].data['img_b_path'] == str(unpair_folder /
                                                         'testB' / '6.jpg'))

        # input path is str
        generation_unpaired_dataset = GenerationUnpairedDataset(
            dataroot=str(unpair_folder), pipeline=pipeline, test_mode=True)
        data_infos_a = generation_unpaired_dataset.data_infos_a
        data_infos_b = generation_unpaired_dataset.data_infos_b
        assert data_infos_a == [
            dict(path=str(unpair_folder / 'testA' / '5.jpg'))
        ]
        assert data_infos_b == [
            dict(path=str(unpair_folder / 'testB' / '6.jpg'))
        ]
        result = generation_unpaired_dataset[0]
        assert (len(generation_unpaired_dataset) == 1)
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'testA' / '5.jpg'))
        assert (result['meta'].data['img_b_path'] == str(unpair_folder /
                                                         'testB' / '6.jpg'))

        # test_mode = False
        generation_unpaired_dataset = GenerationUnpairedDataset(
            dataroot=str(unpair_folder), pipeline=pipeline, test_mode=False)
        data_infos_a = generation_unpaired_dataset.data_infos_a
        data_infos_b = generation_unpaired_dataset.data_infos_b
        assert data_infos_a == [
            dict(path=str(unpair_folder / 'trainA' / '1.jpg')),
            dict(path=str(unpair_folder / 'trainA' / '2.jpg'))
        ]
        assert data_infos_b == [
            dict(path=str(unpair_folder / 'trainB' / '3.jpg')),
            dict(path=str(unpair_folder / 'trainB' / '4.jpg'))
        ]
        assert (len(generation_unpaired_dataset) == 2)
        img_b_paths = [
            str(unpair_folder / 'trainB' / '3.jpg'),
            str(unpair_folder / 'trainB' / '4.jpg')
        ]
        result = generation_unpaired_dataset[0]
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'trainA' / '1.jpg'))
        assert result['meta'].data['img_b_path'] in img_b_paths
        result = generation_unpaired_dataset[1]
        assert assert_dict_has_keys(result, target_keys)
        assert assert_dict_has_keys(result['meta'].data, target_meta_keys)
        assert (result['meta'].data['img_a_path'] == str(unpair_folder /
                                                         'trainA' / '2.jpg'))
        assert result['meta'].data['img_b_path'] in img_b_paths
