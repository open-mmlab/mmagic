# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.wgan_gp import WGANGPGenerator
from mmagic.registry import MODELS


class TestWGANGPGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((2, 100))
        cls.default_config = dict(
            type='WGANGPGenerator', noise_size=128, out_scale=128)

    def test_wgangp_generator(self):

        # test default setting with builder
        g = MODELS.build(self.default_config)
        assert isinstance(g, WGANGPGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 128, 128)

        # test different out_scale
        config = dict(type='WGANGPGenerator', noise_size=128, out_scale=64)
        g = MODELS.build(config)
        assert isinstance(g, WGANGPGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 64, 64)

        # test different conv config
        config = dict(
            type='WGANGPGenerator',
            noise_size=128,
            out_scale=128,
            conv_module_cfg=dict(
                conv_cfg=None,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                norm_cfg=dict(type='BN'),
                order=('conv', 'norm', 'act')))
        g = MODELS.build(config)
        assert isinstance(g, WGANGPGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 128, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_wgangp_generator_cuda(self):

        # test default setting with builder
        g = MODELS.build(self.default_config).cuda()
        assert isinstance(g, WGANGPGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 128, 128)

        # test different out_scale
        config = dict(type='WGANGPGenerator', noise_size=128, out_scale=64)
        g = MODELS.build(config).cuda()
        assert isinstance(g, WGANGPGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 64, 64)

        # test different conv config
        config = dict(
            type='WGANGPGenerator',
            noise_size=128,
            out_scale=128,
            conv_module_cfg=dict(
                conv_cfg=None,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                norm_cfg=dict(type='BN'),
                order=('conv', 'norm', 'act')))
        g = MODELS.build(config).cuda()
        assert isinstance(g, WGANGPGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 128, 128)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
