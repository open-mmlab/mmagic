# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy
from unittest import TestCase

import pytest
import torch

from mmagic.models.editors.mspie.mspie_stylegan2_modules import (
    ModulatedPEConv2d, ModulatedPEStyleConv)


class TestModulatedPEStyleConv(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cfg = dict(
            in_channels=8, out_channels=8, kernel_size=3, style_channels=16)

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_upsample(self):
        cfg = deepcopy(self.default_cfg)
        conv = ModulatedPEStyleConv(**cfg, upsample=True)
        x = torch.randn(1, 8, 32, 32)
        style = torch.randn(1, 16)
        out = conv(x, style)
        self.assertEqual(out.shape, (1, 8, 64, 64))

        # test return noise
        noise = torch.randn(1, 8, 64, 64)
        out = conv(x, style, noise)
        self.assertEqual(out.shape, (1, 8, 64, 64))

        out, noise_return = conv(x, style, noise, return_noise=True)
        assert (noise_return == noise).all()

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_downsample(self):

        cfg = deepcopy(self.default_cfg)
        conv = ModulatedPEStyleConv(**cfg, downsample=True)
        x = torch.randn(1, 8, 32, 32)
        style = torch.randn(1, 16)
        out = conv(x, style)
        self.assertEqual(out.shape, (1, 8, 16, 16))

        # test return noise
        noise = torch.randn(1, 8, 16, 16)
        out = conv(x, style, noise)
        self.assertEqual(out.shape, (1, 8, 16, 16))

        out, noise_return = conv(x, style, noise, return_noise=True)
        assert (noise_return == noise).all()


class TestModulatedPEConv2d(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cfg = dict(
            in_channels=8, out_channels=8, kernel_size=3, style_channels=16)

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_equalized_lr_cfg(self):
        cfg = deepcopy(self.default_cfg)
        conv = ModulatedPEConv2d(**cfg, equalized_lr_cfg=None)
        x = torch.randn(1, 8, 32, 32)
        style = torch.randn(1, 16)
        out = conv(x, style)
        self.assertEqual(out.shape, (1, 8, 32, 32))

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_demodulate(self):
        cfg = deepcopy(self.default_cfg)
        conv = ModulatedPEConv2d(**cfg, demodulate=False)
        x = torch.randn(1, 8, 32, 32)
        style = torch.randn(1, 16)
        out = conv(x, style)
        self.assertEqual(out.shape, (1, 8, 32, 32))

    @pytest.mark.skipif(
        'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason='skip on windows due to uncompiled ops.')
    def test_up_after_conv(self):
        x = torch.randn(1, 8, 32, 32)
        style = torch.randn(1, 16)
        cfg = deepcopy(self.default_cfg)

        conv = ModulatedPEConv2d(
            **cfg, upsample=True, deconv2conv=True, up_after_conv=True)
        out = conv(x, style)
        self.assertEqual(out.shape, (1, 8, 64, 64))

        conv = ModulatedPEConv2d(
            **cfg, upsample=True, deconv2conv=True, up_after_conv=False)
        out = conv(x, style)
        self.assertEqual(out.shape, (1, 8, 64, 64))

        conv = ModulatedPEConv2d(
            **cfg, upsample=True, deconv2conv=True, interp_pad=3)
        out = conv(x, style)
        self.assertEqual(out.shape, (1, 8, 67, 67))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
