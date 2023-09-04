# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from mmagic.models.editors.pggan import (EqualizedLR,
                                         EqualizedLRConvDownModule,
                                         EqualizedLRConvModule,
                                         EqualizedLRConvUpModule,
                                         EqualizedLRLinearModule,
                                         MiniBatchStddevLayer,
                                         PGGANNoiseTo2DFeat, PixelNorm,
                                         equalized_lr)
from mmagic.utils import register_all_modules

register_all_modules()


class TestEqualizedLR:

    @classmethod
    def setup_class(cls):
        cls.default_conv_cfg = dict(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'))
        cls.conv_input = torch.randn((2, 1, 5, 5))
        cls.linear_input = torch.randn((2, 2))

    def test_equalized_conv_module(self):
        conv = EqualizedLRConvModule(**self.default_conv_cfg)
        res = conv(self.conv_input)
        assert res.shape == (2, 1, 5, 5)
        has_equalized_lr = False
        for _, v in conv.conv._forward_pre_hooks.items():
            if isinstance(v, EqualizedLR):
                has_equalized_lr = True
        assert has_equalized_lr

        conv = EqualizedLRConvModule(
            equalized_lr_cfg=None, **self.default_conv_cfg)
        res = conv(self.conv_input)
        assert res.shape == (2, 1, 5, 5)
        has_equalized_lr = False
        for _, v in conv.conv._forward_pre_hooks.items():
            if isinstance(v, EqualizedLR):
                has_equalized_lr = True
        assert not has_equalized_lr

        conv = EqualizedLRConvModule(
            equalized_lr_cfg=dict(gain=1), **self.default_conv_cfg)
        res = conv(self.conv_input)
        assert res.shape == (2, 1, 5, 5)
        has_equalized_lr = False
        for _, v in conv.conv._forward_pre_hooks.items():
            if isinstance(v, EqualizedLR):
                assert v.gain == 1
                has_equalized_lr = True
        assert has_equalized_lr

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_equalized_conv_module_cuda(self):
        conv = EqualizedLRConvModule(**self.default_conv_cfg).cuda()
        res = conv(self.conv_input.cuda())
        assert res.shape == (2, 1, 5, 5)
        has_equalized_lr = False
        for _, v in conv.conv._forward_pre_hooks.items():
            if isinstance(v, EqualizedLR):
                has_equalized_lr = True
        assert has_equalized_lr

    def test_equalized_linear_module(self):
        linear = EqualizedLRLinearModule(2, 2)
        res = linear(self.linear_input)
        assert res.shape == (2, 2)
        has_equalized_lr = False
        for _, v in linear._forward_pre_hooks.items():
            if isinstance(v, EqualizedLR):
                has_equalized_lr = True
        assert has_equalized_lr

        linear = EqualizedLRLinearModule(2, 2, equalized_lr_cfg=None)
        res = linear(self.linear_input)
        assert res.shape == (2, 2)
        has_equalized_lr = False
        for _, v in linear._forward_pre_hooks.items():
            if isinstance(v, EqualizedLR):
                has_equalized_lr = True
        assert not has_equalized_lr

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_equalized_linear_module_cuda(self):
        linear = EqualizedLRLinearModule(2, 2).cuda()
        res = linear(self.linear_input.cuda())
        assert res.shape == (2, 2)
        has_equalized_lr = False
        for _, v in linear._forward_pre_hooks.items():
            if isinstance(v, EqualizedLR):
                has_equalized_lr = True
        assert has_equalized_lr

    def test_equalized_lr(self):
        with pytest.raises(RuntimeError):
            conv = nn.Conv2d(1, 1, 3, 1, 1)
            conv = equalized_lr(conv)
            conv = equalized_lr(conv)


class TestEqualizedLRConvUpModule:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=2,
            conv_cfg=dict(type='deconv'),
            upsample=dict(type='fused_nn'),
            norm_cfg=dict(type='PixelNorm'))
        cls.default_input = torch.randn((2, 3, 5, 5))

    def test_equalized_lr_convup_module(self, ):
        convup = EqualizedLRConvUpModule(**self.default_cfg)

        res = convup(self.default_input)
        assert res.shape == (2, 1, 10, 10)
        # test bp
        res = convup(torch.randn((2, 3, 5, 5), requires_grad=True))
        assert res.shape == (2, 1, 10, 10)
        res.mean().backward()

        # test nearest
        cfg_ = deepcopy(self.default_cfg)
        cfg_['upsample'] = dict(type='nearest', scale_factor=2)
        cfg_['kernel_size'] = 4
        convup = EqualizedLRConvUpModule(**cfg_)

        res = convup(self.default_input)
        assert res.shape == (2, 1, 20, 20)

        # test nearest
        cfg_ = deepcopy(self.default_cfg)
        cfg_['upsample'] = None
        cfg_['kernel_size'] = 4
        convup = EqualizedLRConvUpModule(**cfg_)

        res = convup(self.default_input)
        assert res.shape == (2, 1, 10, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_equalized_lr_convup_module_cuda(self):
        convup = EqualizedLRConvUpModule(**self.default_cfg).cuda()

        res = convup(self.default_input.cuda())
        assert res.shape == (2, 1, 10, 10)
        # test bp
        res = convup(torch.randn((2, 3, 5, 5), requires_grad=True).cuda())
        assert res.shape == (2, 1, 10, 10)
        res.mean().backward()


class TestEqualizedLRConvDownModule:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=2,
            downsample=dict(type='fused_pool'))
        cls.default_input = torch.randn((2, 3, 8, 8))

    def test_equalized_lr_conv_down(self):
        convdown = EqualizedLRConvDownModule(**self.default_cfg)
        res = convdown(self.default_input)
        assert res.shape == (2, 1, 4, 4)
        # test bp
        res = convdown(torch.randn((2, 3, 8, 8), requires_grad=True))
        assert res.shape == (2, 1, 4, 4)
        res.mean().backward()

        # test avg pool
        cfg_ = deepcopy(self.default_cfg)
        cfg_['downsample'] = dict(type='avgpool', kernel_size=2, stride=2)
        convdown = EqualizedLRConvDownModule(**cfg_)
        res = convdown(self.default_input)
        assert res.shape == (2, 1, 2, 2)

        # test downsample is None
        cfg_ = deepcopy(self.default_cfg)
        cfg_['downsample'] = None
        convdown = EqualizedLRConvDownModule(**cfg_)
        res = convdown(self.default_input)
        assert res.shape == (2, 1, 4, 4)

        with pytest.raises(NotImplementedError):
            cfg_ = deepcopy(self.default_cfg)
            cfg_['downsample'] = dict(type='xxx', kernel_size=2, stride=2)
            _ = EqualizedLRConvDownModule(**cfg_)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_equalized_lr_conv_down_cuda(self):
        convdown = EqualizedLRConvDownModule(**self.default_cfg).cuda()
        res = convdown(self.default_input.cuda())
        assert res.shape == (2, 1, 4, 4)
        # test bp
        res = convdown(torch.randn((2, 3, 8, 8), requires_grad=True).cuda())
        assert res.shape == (2, 1, 4, 4)
        res.mean().backward()


class TestPixelNorm:

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((2, 3, 4, 4))

    def test_pixel_norm(self):
        pn = PixelNorm()
        res = pn(self.input_tensor)
        assert res.shape == (2, 3, 4, 4)

        # test zero case
        res = pn(self.input_tensor * 0)
        assert res.shape == (2, 3, 4, 4)
        assert (res == 0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pixel_norm_cuda(self):
        pn = PixelNorm().cuda()
        res = pn(self.input_tensor.cuda())
        assert res.shape == (2, 3, 4, 4)

        # test zero case
        res = pn(self.input_tensor.cuda() * 0)
        assert res.shape == (2, 3, 4, 4)
        assert (res == 0).all()


class TestMiniBatchStddevLayer:

    @classmethod
    def setup_class(cls):
        cls.default_input = torch.randn((2, 3, 4, 4))

    def test_minibatch_stddev_layer(self):
        ministd_layer = MiniBatchStddevLayer()
        res = ministd_layer(self.default_input)
        assert res.shape == (2, 4, 4, 4)

        with pytest.raises(AssertionError):
            _ = ministd_layer(torch.randn((5, 4, 3, 3)))

        ministd_layer = MiniBatchStddevLayer(group_size=3)
        res = ministd_layer(torch.randn((2, 6, 4, 4)))
        assert res.shape == (2, 7, 4, 4)

        # test bp
        ministd_layer = MiniBatchStddevLayer()
        res = ministd_layer(self.default_input.requires_grad_())
        assert res.shape == (2, 4, 4, 4)
        res.mean().backward()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_minibatch_stddev_layer_cuda(self):
        ministd_layer = MiniBatchStddevLayer().cuda()
        res = ministd_layer(self.default_input.cuda())
        assert res.shape == (2, 4, 4, 4)

        ministd_layer = MiniBatchStddevLayer(group_size=3).cuda()
        res = ministd_layer(torch.randn((2, 6, 4, 4)).cuda())
        assert res.shape == (2, 7, 4, 4)

        # test bp
        ministd_layer = MiniBatchStddevLayer().cuda()
        res = ministd_layer(self.default_input.requires_grad_().cuda())
        assert res.shape == (2, 4, 4, 4)
        res.mean().backward()


class TestPGGANNoiseTo2DFeat:

    @classmethod
    def setup_class(cls):
        cls.default_input = torch.randn((2, 10))
        cls.default_cfg = dict(noise_size=10, out_channels=1)

    def test_pggan_noise2feat(self):
        module = PGGANNoiseTo2DFeat(**self.default_cfg)
        res = module(self.default_input)
        assert res.shape == (2, 1, 4, 4)
        assert isinstance(module.linear, EqualizedLRLinearModule)
        assert not module.linear.bias
        assert module.with_norm
        assert isinstance(module.norm, PixelNorm)
        assert isinstance(module.activation, nn.LeakyReLU)

        module = PGGANNoiseTo2DFeat(**self.default_cfg, act_cfg=None)
        res = module(self.default_input)
        assert res.shape == (2, 1, 4, 4)
        assert isinstance(module.linear, EqualizedLRLinearModule)
        assert not module.linear.bias
        assert module.with_norm
        assert not module.with_activation

        module = PGGANNoiseTo2DFeat(
            **self.default_cfg, norm_cfg=None, normalize_latent=False)
        res = module(self.default_input)
        assert res.shape == (2, 1, 4, 4)
        assert isinstance(module.linear, EqualizedLRLinearModule)
        assert not module.linear.bias
        assert not module.with_norm
        assert isinstance(module.activation, nn.LeakyReLU)

        with pytest.raises(AssertionError):
            _ = module(torch.randn((2, 1, 2)))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pggan_noise2feat_cuda(self):
        module = PGGANNoiseTo2DFeat(**self.default_cfg).cuda()
        res = module(self.default_input.cuda())
        assert res.shape == (2, 1, 4, 4)
        assert isinstance(module.linear, EqualizedLRLinearModule)
        assert not module.linear.bias
        assert module.with_norm
        assert isinstance(module.activation, nn.LeakyReLU)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
