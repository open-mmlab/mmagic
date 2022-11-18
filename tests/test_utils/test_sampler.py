# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.utils.sampler import ArgumentsSampler


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
