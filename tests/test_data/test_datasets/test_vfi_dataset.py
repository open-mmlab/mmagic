# Copyright (c) OpenMMLab. All rights reserved.
import pytest
from mmcv.utils.testing import assert_dict_has_keys

from mmedit.datasets import BaseVFIDataset, build_dataset


class TestVFIDataset:

    pipeline = [
        dict(type='LoadImageFromFileList', io_backend='disk', key='inputs'),
        dict(type='LoadImageFromFile', io_backend='disk', key='target'),
        dict(type='FramesToTensor', keys=['inputs']),
        dict(type='ImageToTensor', keys=['target']),
    ]
    folder = 'tests/data/vimeo90k'
    ann_file = 'tests/data/vimeo90k/vfi_ann.txt'

    def test_base_vfi_dataset(self):

        dataset = BaseVFIDataset(self.pipeline, self.folder, self.ann_file)
        dataset.__init__(self.pipeline, self.folder, self.ann_file)
        dataset.load_annotations()
        assert dataset.folder == self.folder
        assert dataset.ann_file == self.ann_file
        setattr(dataset, 'data_infos', [
            dict(
                inputs_path=[
                    'tests/data/vimeo90k/00001/0266/im1.png',
                    'tests/data/vimeo90k/00001/0266/im3.png'
                ],
                target_path='tests/data/vimeo90k/00001/0266/im2.png',
                key='00001/0266')
        ])
        data = dataset.__getitem__(0)
        assert_dict_has_keys(data, ['folder', 'ann_file'])
        results = [dict(eval_result=dict(psnr=1.1, ssim=0.3))]
        eval_result = dataset.evaluate(results)
        assert_dict_has_keys(eval_result, ['psnr', 'ssim'])

        with pytest.raises(TypeError):
            dataset.evaluate(results[0])
        with pytest.raises(AssertionError):
            dataset.evaluate(results + results)

    def test_vfi_vimeo90k_dataset(self):

        dataset_cfg = dict(
            type='VFIVimeo90KDataset',
            folder=self.folder,
            ann_file=self.ann_file,
            pipeline=self.pipeline)
        dataset = build_dataset(dataset_cfg)
        data_infos = dataset.data_infos[0]
        assert_dict_has_keys(data_infos, ['inputs_path', 'target_path', 'key'])


def test_vfi_dataset():
    test_ = TestVFIDataset()
    test_.test_base_vfi_dataset()
    test_.test_vfi_vimeo90k_dataset()
