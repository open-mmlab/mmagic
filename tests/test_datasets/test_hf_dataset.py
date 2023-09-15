# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.testing import RunnerTestCase

from mmagic.datasets import HuggingFaceDataset


class TestHFDataset(RunnerTestCase):

    def test_dataset_from_local(self):
        dataset = HuggingFaceDataset(
            dataset='tests/data/sd', image_column='file_name')
        assert len(dataset) == 1

        data = dataset[0]
        assert data['prompt'] == 'a dog'
        assert data['img'] == 'tests/data/sd/color.jpg'

        dataset = HuggingFaceDataset(
            dataset='tests/data/sd',
            image_column='file_name',
            csv='metadata2.csv')
        assert len(dataset) == 1

        data = dataset[0]
        assert data['prompt'] == 'a cat'
        assert data['img'] == 'tests/data/sd/color.jpg'
