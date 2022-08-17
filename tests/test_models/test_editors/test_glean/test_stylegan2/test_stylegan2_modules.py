# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmedit.models.editors.glean.stylegan2.stylegan2_modules import (
    Blur, ModulatedStyleConv, ModulatedToRGB)


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
