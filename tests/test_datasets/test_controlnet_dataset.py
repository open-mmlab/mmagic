# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmagic.datasets import ControlNetDataset

data_dir = os.path.join(__file__, '../', '../', 'data', 'controlnet')
data_dir = os.path.abspath(data_dir)
anno_path = os.path.join(data_dir, 'prompt.json')
source_path = os.path.join(data_dir, 'source')
target_path = os.path.join(data_dir, 'target')


def test_controlnet_dataset():
    print(os.path.abspath(data_dir))
    dataset = ControlNetDataset(data_root=data_dir)
    assert len(dataset) == 2
    prompts = [
        'pale golden rod circle with old lace background',
        'light coral circle with white background'
    ]
    for idx, data in enumerate(dataset):
        assert 'source_path' in data
        assert 'target_path' in data
        assert data['prompt'] == prompts[idx]


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
