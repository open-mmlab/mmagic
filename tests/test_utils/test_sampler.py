# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from mmagic.utils.sampler import ArgumentsSampler, ValDataSampler


def test_argument_sampler():
    sample_kwargs = dict(
        a=1,
        b=2,
        max_times=10,
        num_batches=2,
        forward_kwargs=dict(forward_mode='gen'))
    sampler = ArgumentsSampler(sample_kwargs=sample_kwargs, )

    assert sampler.max_times == 10
    for sample in sampler:
        assert 'inputs' in sample
        assert sample['inputs'] == dict(forward_mode='gen', num_batches=2)


class MockDataset():

    def __init__(self, length):
        self.length = length

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.length


class MockValLoop():

    def __init__(self):
        self.dataloaders = None


def test_val_data_sampler():
    runner = MagicMock()
    val_loop = MockValLoop()
    val_loop.dataloaders = [
        DataLoader(MockDataset(10), batch_size=4),
        DataLoader(MockDataset(5), batch_size=4)
    ]
    # val_loop.dataloader = None
    runner.val_loop = val_loop

    val_sampler = ValDataSampler(
        sample_kwargs=dict(max_times=10), runner=runner)
    assert len(val_sampler._dataloader.dataset) == 15
    tar_out = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1], [2, 3, 4]]
    for idx, out in enumerate(val_sampler):
        assert out == tar_out[idx]

    setattr(val_loop, 'dataloader', DataLoader(MockDataset(8), batch_size=4))
    val_sampler = ValDataSampler(
        sample_kwargs=dict(max_times=10), runner=runner)
    assert len(val_sampler._dataloader.dataset) == 8
    for idx, out in enumerate(val_sampler):
        assert out == tar_out[idx]

    # test iteration times
    val_sampler = ValDataSampler(
        sample_kwargs=dict(max_times=1), runner=runner)
    tar_out = [[0, 1, 2, 3]]
    for idx, out in enumerate(val_sampler):
        assert out == tar_out[idx]


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
