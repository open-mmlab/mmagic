# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.mspie import MSStyleGAN2Discriminator


class TestMSStyleGANv2Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64, channel_multiplier=1)

    def test_msstylegan2_disc_cpu(self):
        d = MSStyleGAN2Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

        d = MSStyleGAN2Discriminator(
            with_adaptive_pool=True, **self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_msstylegan2_disc_cuda(self):
        d = MSStyleGAN2Discriminator(**self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)

        d = MSStyleGAN2Discriminator(
            with_adaptive_pool=True, **self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
