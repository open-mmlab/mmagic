# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmagic.datasets import DreamBoothDataset

# we use controlnet's dataset to test
data_dir = os.path.join(__file__, '../', '../', 'data', 'controlnet')
concept_dir = os.path.join(data_dir, 'source')


def test_dreambooth_dataset():
    print(os.path.abspath(data_dir))
    dataset = DreamBoothDataset(
        data_root=data_dir,
        concept_dir=concept_dir,
        prompt='a sks ball',
    )
    assert len(dataset) == 2
    for data in dataset:
        assert data['prompt'] == 'a sks ball'


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
