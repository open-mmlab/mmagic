# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch.nn as nn
from mmengine import MMLogger
from mmengine.optim import OptimWrapper

from mmagic.engine.optimizers import MultiOptimWrapperConstructor

logger = MMLogger.get_instance('test_multi_optimizer_constructor')


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.generator = nn.Conv2d(3, 3, 1)
        self.discriminator = nn.Conv2d(3, 3, 1)


class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 100)


class ToyModel2(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.m1 = ToyModel()
        self.m2 = nn.Conv2d(3, 3, 1)
        self.m3 = nn.Linear(2, 2)
        self.text_encoder = TextEncoder()


def test_optimizer_constructor():

    # test optimizer wrapper cfg is a dict
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

    # test optimizer wrapper is dict of **modules**
    optim_wrapper = {
        '.*embedding': {
            'type': 'OptimWrapper',
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-4,
                'betas': (0.9, 0.99)
            }
        },
        'm1.generator': {
            'type': 'OptimWrapper',
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-5,
                'betas': (0.9, 0.99)
            }
        },
        'm2': {
            'type': 'OptimWrapper',
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-5,
            }
        }
    }
    optim_wrapper_constructor = MultiOptimWrapperConstructor(
        optim_wrapper_cfg=dict(
            generator=dict(
                type='OptimWrapper',
                optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))),
            discriminator=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.1))))

    optim_wrapper_cfg = {
        '.*embedding': {
            'type': 'OptimWrapper',
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-4,
                'betas': (0.9, 0.99)
            }
        },
        'm1.generator': {
            'type': 'OptimWrapper',
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-5,
                'betas': (0.9, 0.99)
            }
        },
        'm2': {
            'type': 'OptimWrapper',
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-5,
            }
        }
    }
    optim_wrapper_constructor = MultiOptimWrapperConstructor(optim_wrapper_cfg)
    model = ToyModel2()
    optim_wrapper_dict = optim_wrapper_constructor(model)

    # optim_wrapper_cfg should be a dict
    with pytest.raises(TypeError):
        MultiOptimWrapperConstructor(1)

    # parawise_cfg should be set in each optimizer separately
    with pytest.raises(AssertionError):
        MultiOptimWrapperConstructor(dict(), dict())

    # test optimizer wrapper with multi param groups
    optim_wrapper_constructor = MultiOptimWrapperConstructor(
        optim_wrapper_cfg=dict(
            modules=['.*text_encoder', '.*generator', 'm2'],
            optimizer=dict(
                type='Adam',
                lr=1e-4,
                betas=(0.9, 0.99),
            ),
            accumulative_counts=4))
    model = ToyModel2()
    optim_wrapper = optim_wrapper_constructor(model)
    assert isinstance(optim_wrapper, OptimWrapper)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
