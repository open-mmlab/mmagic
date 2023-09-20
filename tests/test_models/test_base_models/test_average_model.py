# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import pytest
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.testing import assert_allclose

from mmagic.models.base_models import ExponentialMovingAverage, RampUpEMA


class ToyModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.register_buffer('buffer', torch.randn(2, 2))

    def update_buffer(self):
        for buffer in self.buffers():
            buffer.add_(torch.randn(*buffer.size()))

    def update_params(self):
        for param in self.parameters():
            param.add(torch.randn(*param.size()))

    def update(self):
        self.update_params()
        self.update_buffer()

    def forward(self, x):
        return self.conv(x)


class ToyModel_old(BaseModel):

    def __init__(self):
        super().__init__()
        self.ema = ToyModule()
        self.module = ToyModule()

    def forward(self, x):
        return


class ToyEMAModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ToyModule()

    def forward(self, x):
        return self.module(x)


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.ema = ToyEMAModel()
        self.module = ToyModule()

    def forward(self, x):
        return


class TestExponentialMovingAverage(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cfg = dict(
            interval=1, momentum=0.0001, update_buffers=True)

    def test_init(self):
        cfg = deepcopy(self.default_cfg)
        model = ToyModule()
        average_model = ExponentialMovingAverage(model=model, **cfg)
        self.assertEqual(average_model.momentum, 0.0001)
        self.assertTrue(
            (average_model.module.conv.weight == model.conv.weight).all())

        cfg['momentum'] = 10
        with self.assertRaises(AssertionError):
            average_model = ExponentialMovingAverage(model, **cfg)

        cfg['momentum'] = -2
        with self.assertRaises(AssertionError):
            average_model = ExponentialMovingAverage(model, **cfg)

        # test warning
        with pytest.warns(UserWarning):
            cfg = deepcopy(self.default_cfg)
            cfg['momentum'] = 0.6
            model = ToyModule()
            average_model = ExponentialMovingAverage(model=model, **cfg)
            self.assertEqual(average_model.momentum, 0.6)

    def test_avg_func(self):
        cfg = deepcopy(self.default_cfg)
        model = ToyModule()
        average_model = ExponentialMovingAverage(model, **cfg)
        src_tensor = torch.randn(1, 3, 2, 2)
        tar_tensor = torch.randn(1, 3, 2, 2)
        tar_tensor_backup = tar_tensor.clone()
        average_model.avg_func(tar_tensor, src_tensor, steps=42)
        assert_allclose(tar_tensor,
                        tar_tensor_backup * 0.9999 + src_tensor * 0.0001)

    def test_sync_buffer_and_parameters(self):
        cfg = deepcopy(self.default_cfg)
        model = ToyModule()
        average_model = ExponentialMovingAverage(model, **cfg)
        model.update()
        average_model.sync_parameters(model)

        assert_allclose(model.conv.weight, average_model.module.conv.weight)
        assert_allclose(model.conv.bias, average_model.module.conv.bias)
        assert_allclose(model.buffer, average_model.module.buffer)

        model.update()
        average_model.sync_buffers(model)
        assert_allclose(model.buffer, average_model.module.buffer)

        cfg['update_buffers'] = True
        average_model = ExponentialMovingAverage(model, **cfg)
        with self.assertWarns(Warning):
            average_model.sync_buffers(model)

    def test_load_from_state_dict(self):
        cfg = deepcopy(self.default_cfg)

        model = ToyModel()
        average_model = ExponentialMovingAverage(model, **cfg)

        old_state_dict = ToyModel_old().state_dict()
        average_model._load_from_state_dict(
            old_state_dict,
            'ema.',
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[])


class TestRamUpEMA(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.default_cfg = dict(
            interval=1, ema_kimg=10, ema_rampup=0.05, batch_size=32, eps=1e-8)

    def test_init(self):
        cfg = deepcopy(self.default_cfg)
        model = ToyModule()
        RampUpEMA(model, **cfg)

    def test_avg_func(self):
        cfg = deepcopy(self.default_cfg)
        model = ToyModule()
        average_model = RampUpEMA(model, **cfg)
        src_tensor = torch.randn(1, 3, 2, 2)
        tar_tensor = torch.randn(1, 3, 2, 2)
        average_model.avg_func(tar_tensor, src_tensor, steps=42)

        # model = ToyModule()
        cfg['ema_rampup'] = None
        average_model = RampUpEMA(model, **cfg)
        src_tensor = torch.randn(1, 3, 2, 2)
        tar_tensor = torch.randn(1, 3, 2, 2)
        average_model.avg_func(tar_tensor, src_tensor, steps=42)

        # test warning
        with pytest.warns(UserWarning):
            cfg = deepcopy(self.default_cfg)
            cfg['batch_size'] = 0
            model = ToyModule()
            average_model = RampUpEMA(model, **cfg)
            src_tensor = torch.randn(1, 3, 2, 2)
            tar_tensor = torch.randn(1, 3, 2, 2)
            average_model.avg_func(tar_tensor, src_tensor, steps=42)

    def test_sync_buffer_and_parameters(self):
        cfg = deepcopy(self.default_cfg)
        model = ToyModule()
        average_model = RampUpEMA(model, **cfg)
        model.update()
        average_model.sync_parameters(model)

        assert_allclose(model.conv.weight, average_model.module.conv.weight)
        assert_allclose(model.conv.bias, average_model.module.conv.bias)
        assert_allclose(model.buffer, average_model.module.buffer)

        model.update()
        average_model.sync_buffers(model)
        assert_allclose(model.buffer, average_model.module.buffer)

        cfg['update_buffers'] = True
        average_model = RampUpEMA(model, **cfg)
        with self.assertWarns(Warning):
            average_model.sync_buffers(model)

    def test_load_from_state_dict(self):
        cfg = deepcopy(self.default_cfg)
        model = ToyModel()
        average_model = RampUpEMA(model, **cfg)

        old_state_dict = ToyModel_old().state_dict()
        average_model._load_from_state_dict(
            old_state_dict,
            'ema.',
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[])


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
