# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.editors.lsgan import LSGANDiscriminator
from mmedit.registry import MODULES


class TestLSGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.x = torch.randn((2, 3, 128, 128))
        cls.default_config = dict(
            type='LSGANDiscriminator', in_channels=3, input_scale=128)

    def test_lsgan_discriminator(self):

        # test default setting with builder
        d = MODULES.build(self.default_config)
        assert isinstance(d, LSGANDiscriminator)
        score = d(self.x)
        assert score.shape == (2, 1)

        # test different input_scale
        config = dict(type='LSGANDiscriminator', in_channels=3, input_scale=64)
        d = MODULES.build(config)
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x)
        assert score.shape == (2, 1)

        # test different config
        config = dict(
            type='LSGANDiscriminator',
            in_channels=3,
            input_scale=64,
            out_act_cfg=dict(type='Sigmoid'))
        d = MODULES.build(config)
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_lsgan_discriminator_cuda(self):

        # test default setting with builder
        d = MODULES.build(self.default_config).cuda()
        assert isinstance(d, LSGANDiscriminator)
        score = d(self.x.cuda())
        assert score.shape == (2, 1)

        # test different input_scale
        config = dict(type='LSGANDiscriminator', in_channels=3, input_scale=64)
        d = MODULES.build(config).cuda()
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x.cuda())
        assert score.shape == (2, 1)

        # test different config
        config = dict(
            type='LSGANDiscriminator',
            in_channels=3,
            input_scale=64,
            out_act_cfg=dict(type='Sigmoid'))
        d = MODULES.build(config).cuda()
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x.cuda())
        assert score.shape == (2, 1)
