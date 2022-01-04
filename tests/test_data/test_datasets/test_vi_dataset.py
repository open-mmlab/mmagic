import pytest
from mmcv.utils.testing import assert_dict_has_keys

from mmedit.datasets import BaseVIDataset


class TestVIDataset:

    def __init__(self):
        self.pipeline = [
            dict(
                type='LoadImageFromFileList', io_backend='disk', key='inputs'),
            dict(type='LoadImageFromFile', io_backend='disk', key='target'),
            dict(type='FramesToTensor', keys=['inputs']),
            dict(type='ImageToTensor', keys=['target']),
        ]
        self.folder = 'tests/data/vimeo90k'
        self.ann_file = 'tests/data/vimeo90k/vi_ann.txt'

    def test_base_vi_dataset(self):

        dataset = BaseVIDataset(self.pipeline, self.folder, self.ann_file)
        setattr(dataset, 'data_infos', [
            dict(
                inputs_path=[
                    'tests/data/vimeo90k/00001/0266/im1.png',
                    'tests/data/vimeo90k/00001/0266/im3.png'
                ],
                target_path='tests/data/vimeo90k/00001/0266/im2.png',
                key='00001/0266')
        ])
        dataset.__getitem__(0)
        results = [dict(eval_result=dict(psnr=1.1, ssim=0.3))]
        eval_result = dataset.evaluate(results)
        assert_dict_has_keys(eval_result, ['psnr', 'ssim'])

        with pytest.raises(TypeError):
            dataset.evaluate(results[0])
        with pytest.raises(AssertionError):
            dataset.evaluate(results + results)
