# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch
from mmengine import Config

from mmedit.apis import sample_conditional_model, sample_unconditional_model
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules

register_all_modules()


def test_unconditional_inference():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', 'configs', 'dcgan',
        'dcgan_Glr4e-4_Dlr1e-4_1xb128-5kiters_mnist-64x64.py')
    cfg = Config.fromfile(cfg)
    model = MODELS.build(cfg.model)
    model.eval()

    # test num_samples can be divided by num_batches
    results = sample_unconditional_model(
        model, num_samples=4, sample_model='orig')
    assert results.shape == (4, 1, 64, 64)

    # test num_samples can not be divided by num_batches
    results = sample_unconditional_model(
        model, num_samples=4, num_batches=3, sample_model='orig')
    assert results.shape == (4, 1, 64, 64)


def test_conditional_inference():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', 'configs', 'sngan_proj',
        'sngan-proj_woReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py')
    cfg = Config.fromfile(cfg)
    model = MODELS.build(cfg.model)
    model.eval()

    # test label is int
    results = sample_conditional_model(
        model, label=1, num_samples=4, sample_model='orig')
    assert results.shape == (4, 3, 32, 32)
    # test label is tensor
    results = sample_conditional_model(
        model,
        label=torch.FloatTensor([1.]),
        num_samples=4,
        sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is multi tensor
    results = sample_conditional_model(
        model,
        label=torch.FloatTensor([1., 2., 3., 4.]),
        num_samples=4,
        sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is list of int
    results = sample_conditional_model(
        model, label=[1, 2, 3, 4], num_samples=4, sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is None
    results = sample_conditional_model(
        model, num_samples=4, sample_model='orig')
    assert results.shape == (4, 3, 32, 32)

    # test label is invalid
    with pytest.raises(TypeError):
        results = sample_conditional_model(
            model, label='1', num_samples=4, sample_model='orig')

    # test length of label is not same as num_samples
    with pytest.raises(ValueError):
        results = sample_conditional_model(
            model, label=[1, 2], num_samples=4, sample_model='orig')

    # test num_samples can not be divided by num_batches
    results = sample_conditional_model(
        model, num_samples=3, num_batches=2, sample_model='orig')
    assert results.shape == (3, 3, 32, 32)
