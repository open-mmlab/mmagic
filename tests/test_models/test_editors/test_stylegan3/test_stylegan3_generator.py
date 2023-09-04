# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch

from mmagic.models.editors.stylegan3 import StyleGAN3Generator


class TestStyleGAN3Generator:

    @classmethod
    def setup_class(cls):
        synthesis_cfg = {
            'type': 'SynthesisNetwork',
            'channel_base': 1024,
            'channel_max': 16,
            'magnitude_ema_beta': 0.999
        }
        cls.default_cfg = dict(
            noise_size=6,
            style_channels=8,
            out_size=16,
            img_channels=3,
            synthesis_cfg=synthesis_cfg)
        synthesis_r_cfg = {
            'type': 'SynthesisNetwork',
            'channel_base': 1024,
            'channel_max': 16,
            'magnitude_ema_beta': 0.999,
            'conv_kernel': 1,
            'use_radial_filters': True
        }
        cls.s3_r_cfg = dict(
            noise_size=6,
            style_channels=8,
            out_size=16,
            img_channels=3,
            synthesis_cfg=synthesis_r_cfg)

    @pytest.mark.skipif(
        ('win' in platform.system().lower() and 'cu' in torch.__version__)
        or not torch.cuda.is_available(),
        reason='skip on windows-cuda due to limited RAM.')
    def test_cpu(self):
        generator = StyleGAN3Generator(**self.default_cfg)
        z = torch.randn((2, 6))
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)

        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(rgb2bgr=True))
        generator = StyleGAN3Generator(**cfg)
        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        # test return latents
        result = generator(None, num_batches=2, return_latents=True)
        assert isinstance(result, dict)
        assert result['fake_img'].shape == (2, 3, 16, 16)
        assert result['noise_batch'].shape == (2, 6)
        assert result['latent'].shape == (2, 16, 8)

        # test input_is_latent
        result = generator(
            None, num_batches=2, input_is_latent=True, return_latents=True)
        assert isinstance(result, dict)
        assert result['fake_img'].shape == (2, 3, 16, 16)
        assert result['noise_batch'].shape == (2, 8)
        assert result['latent'].shape == (2, 16, 8)

        generator = StyleGAN3Generator(**self.s3_r_cfg)
        z = torch.randn((2, 6))
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        generator = StyleGAN3Generator(**self.default_cfg).cuda()
        z = torch.randn((2, 6)).cuda()
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)

        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(rgb2bgr=True))
        generator = StyleGAN3Generator(**cfg).cuda()
        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        generator = StyleGAN3Generator(**self.s3_r_cfg).cuda()
        z = torch.randn((2, 6)).cuda()
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
