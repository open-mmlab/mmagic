# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch

from mmagic.models.editors.stylegan1 import get_mean_latent, style_mixing
from mmagic.models.editors.stylegan2 import StyleGAN2Generator
from mmagic.models.editors.stylegan2.stylegan2_modules import (
    Blur, DownsampleUpFIRDn, ModulatedConv2d, ModulatedStyleConv,
    ModulatedToRGB)
from mmagic.models.utils import get_module_device


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_get_module_device():
    config = dict(
        out_size=64, style_channels=16, num_mlps=4, channel_multiplier=1)
    g = StyleGAN2Generator(**config)
    res = g(None, num_batches=2)
    assert res.shape == (2, 3, 64, 64)

    truncation_mean = get_mean_latent(g, 4096)
    res = g(
        None,
        num_batches=2,
        randomize_noise=False,
        truncation=0.7,
        truncation_latent=truncation_mean)

    # res = g.style_mixing(2, 2, truncation_latent=truncation_mean)
    res = style_mixing(
        g,
        n_source=2,
        n_target=2,
        truncation_latent=truncation_mean,
        style_channels=g.style_channels)

    assert get_module_device(g) == torch.device('cpu')


class TestDownsampleUpFIRDn():

    def test_DownsampleUpFIRDn(self):
        downsample = DownsampleUpFIRDn((2, 2), 2)
        assert downsample.pad == (0, 0)

        inp = torch.randn(1, 3, 4, 4)
        out = downsample(inp)
        assert out.shape == (1, 3, 2, 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_DownsampleUpFIRDn_cuda(self):
        downsample = DownsampleUpFIRDn((2, 2), 2).cuda()
        assert downsample.pad == (0, 0)

        inp = torch.randn(1, 3, 4, 4).cuda()
        out = downsample(inp)
        assert out.shape == (1, 3, 2, 2)


class TestBlur:

    @classmethod
    def setup_class(cls):
        cls.kernel = [1, 3, 3, 1]
        cls.pad = (1, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_blur_cuda(self):
        blur = Blur(self.kernel, self.pad)
        x = torch.randn((2, 3, 8, 8))
        res = blur(x)

        assert res.shape == (2, 3, 7, 7)


class TestModStyleConv:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            style_channels=5,
            upsample=True)

    def test_mod_styleconv_cpu(self):
        conv = ModulatedStyleConv(**self.default_cfg)
        input_x = torch.randn((2, 3, 4, 4))
        input_style = torch.randn((2, 5))

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 8, 8)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        conv = ModulatedStyleConv(**_cfg)
        input_x = torch.randn((2, 3, 4, 4))
        input_style = torch.randn((2, 5))

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)

        # test add noise
        noise = torch.randn(2, 1, 4, 4)
        res = conv(input_x, input_style, noise)
        assert res.shape == (2, 1, 4, 4)

        # test add noise + return_noise
        res = conv(input_x, input_style, noise, return_noise=True)
        assert isinstance(res, tuple)
        assert res[0].shape == (2, 1, 4, 4)
        assert (res[1] == noise).all()

        # test add noise is False
        res = conv(input_x, input_style, noise, add_noise=False)
        assert res.shape == (2, 1, 4, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_mod_styleconv_cuda(self):
        conv = ModulatedStyleConv(**self.default_cfg).cuda()
        input_x = torch.randn((2, 3, 4, 4)).cuda()
        input_style = torch.randn((2, 5)).cuda()

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 8, 8)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        conv = ModulatedStyleConv(**_cfg).cuda()
        input_x = torch.randn((2, 3, 4, 4)).cuda()
        input_style = torch.randn((2, 5)).cuda()

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)


class TestModulatedConv2d():

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            style_channels=5,
            upsample=True)

    def test_mod_conv_cpu(self):
        conv = ModulatedConv2d(**self.default_cfg)
        input_x = torch.randn((2, 3, 4, 4))
        input_style = torch.randn((2, 5))

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 8, 8)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        conv = ModulatedConv2d(**_cfg)
        input_x = torch.randn((2, 3, 4, 4))
        input_style = torch.randn((2, 5))

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        _cfg['downsample'] = True
        conv = ModulatedConv2d(**_cfg)
        input_x = torch.randn((2, 3, 8, 8))
        input_style = torch.randn((2, 5))
        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)

        # test input gain
        res = conv(input_x, input_style, input_gain=torch.randn(2, 3))
        assert res.shape == (2, 1, 4, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_mod_conv_cuda(self):
        conv = ModulatedConv2d(**self.default_cfg).cuda()
        input_x = torch.randn((2, 3, 4, 4)).cuda()
        input_style = torch.randn((2, 5)).cuda()

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 8, 8)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        conv = ModulatedConv2d(**_cfg).cuda()
        input_x = torch.randn((2, 3, 4, 4)).cuda()
        input_style = torch.randn((2, 5)).cuda()

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        _cfg['downsample'] = True
        conv = ModulatedConv2d(**_cfg).cuda()
        input_x = torch.randn((2, 3, 8, 8)).cuda()
        input_style = torch.randn((2, 5)).cuda()
        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)

        # test input gain
        res = conv(input_x, input_style, input_gain=torch.randn(2, 3).cuda())
        assert res.shape == (2, 1, 4, 4)


class TestToRGB:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_channels=5, style_channels=5, out_channels=3)

    def test_torgb_cpu(self):
        model = ModulatedToRGB(**self.default_cfg)
        input_x = torch.randn((2, 5, 4, 4))
        style = torch.randn((2, 5))

        res = model(input_x, style)
        assert res.shape == (2, 3, 4, 4)

        input_x = torch.randn((2, 5, 8, 8))
        style = torch.randn((2, 5))
        skip = torch.randn(2, 3, 4, 4)
        res = model(input_x, style, skip)
        assert res.shape == (2, 3, 8, 8)

        # test skip is passed + upsample is False
        cfg = deepcopy(self.default_cfg)
        cfg['upsample'] = False
        cfg['out_channels'] = 7
        model = ModulatedToRGB(**cfg)
        input_x = torch.randn((2, 5, 4, 4))
        skip = torch.randn(2, 7, 4, 4)
        style = torch.randn((2, 5))
        res = model(input_x, style, skip)
        assert res.shape == (2, 7, 4, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_torgb_cuda(self):
        model = ModulatedToRGB(**self.default_cfg).cuda()
        input_x = torch.randn((2, 5, 4, 4)).cuda()
        style = torch.randn((2, 5)).cuda()

        res = model(input_x, style)
        assert res.shape == (2, 3, 4, 4)

        input_x = torch.randn((2, 5, 8, 8)).cuda()
        style = torch.randn((2, 5)).cuda()
        skip = torch.randn(2, 3, 4, 4).cuda()
        res = model(input_x, style, skip)
        assert res.shape == (2, 3, 8, 8)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
