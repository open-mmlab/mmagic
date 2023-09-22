# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform

import pytest
from mmengine.testing import RunnerTestCase

from mmagic.datasets import HuggingFaceDataset


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestHFDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        data_root = osp.join(osp.dirname(__file__), '../../')
        dataset_path = data_root + 'tests/data/sd'
        dataset = HuggingFaceDataset(
            dataset=dataset_path, image_column='file_name')
        assert len(dataset) == 1

        data = dataset[0]
        assert data['prompt'] == 'a dog'
        assert 'tests/data/sd/color.jpg' in data['img']

        dataset = HuggingFaceDataset(
            dataset='tests/data/sd',
            image_column='file_name',
            csv='metadata2.csv')
        assert len(dataset) == 1

        data = dataset[0]
        assert data['prompt'] == 'a cat'
        assert 'tests/data/sd/color.jpg' in data['img']


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
