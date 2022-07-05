# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch.nn as nn

from mmedit.optimizer import MultiOptimWrapperConstructor


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.generator = nn.Conv2d(3, 3, 1)
        self.discriminator = nn.Conv2d(3, 3, 1)


def test_optimizer_constructor():

    optim_wrapper_constructor = MultiOptimWrapperConstructor(
        optim_wrapper_cfg=dict(
            generator=dict(
                type='OptimWrapper',
                optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))),
            discriminator=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.1))))
    model = ToyModel()

    optim_wrapper_dict = optim_wrapper_constructor(model)
    assert optim_wrapper_dict.__class__.__name__ == 'OptimWrapperDict'
    assert set(optim_wrapper_dict.optim_wrappers) == set(
        ['generator', 'discriminator'])

    # optim_wrapper_cfg should be a dict
    with pytest.raises(TypeError):
        MultiOptimWrapperConstructor(1)

    # parawise_cfg should be set in each optimizer separately
    with pytest.raises(AssertionError):
        MultiOptimWrapperConstructor(dict(), dict())
