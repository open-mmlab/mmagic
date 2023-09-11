# Copyright (c) OpenMMLab. All rights reserved.
from PIL import Image

from mmagic.datasets import HuggingFaceDreamBoothDataset


def test_dreambooth_dataset():
    dataset = HuggingFaceDreamBoothDataset(
        dataset='google/dreambooth',
        dataset_sub_dir='backpack',
        prompt='a sks backpack',
    )
    assert len(dataset) == 6
    for data in dataset:
        assert data['prompt'] == 'a sks backpack'
        assert isinstance(data['img'], Image.Image)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
