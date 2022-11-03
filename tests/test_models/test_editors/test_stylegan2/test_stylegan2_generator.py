# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch

from mmedit.models.editors.stylegan2 import StyleGAN2Generator


class TestStyleGAN2Generator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            out_size=64, style_channels=16, num_mlps=4, channel_multiplier=1)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_stylegan2_g_cpu(self):
        # test default config
        g = StyleGAN2Generator(**self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        truncation_mean = g.get_mean_latent()
        res = g(
            None,
            num_batches=2,
            randomize_noise=False,
            truncation=0.7,
            truncation_latent=truncation_mean)
        assert res.shape == (2, 3, 64, 64)

        res = g.style_mixing(2, 2, truncation_latent=truncation_mean)
        assert res.shape[2] == 64

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        styles = [torch.randn((1, 16)) for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)) for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 256
        g = StyleGAN2Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        # set mix_prob as 1 and 0 to cover all lines
        g.mix_prob = 1
        res = g(None, num_batches=2)
        g.mix_prob = 0
        res = g(None, num_batches=2)

        # test cond channels is negative number
        cfg_ = deepcopy(self.default_cfg)
        cfg_['cond_channels'] = -1
        g = StyleGAN2Generator(**cfg_)
        assert not hasattr(g, 'embed')

        # test cond channels > 0
        cfg_ = deepcopy(self.default_cfg)
        cfg_['cond_channels'] = 10
        g = StyleGAN2Generator(**cfg_)
        assert hasattr(g, 'embed')
        # test raise error
        with pytest.raises(AssertionError):
            g(None, num_batches=2)
        res = g(None, num_batches=2, cond=torch.randn(2, 10))
        assert res.shape == (2, 3, 64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_g_cuda(self):
        # test default config
        g = StyleGAN2Generator(**self.default_cfg).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        styles = [torch.randn((1, 16)).cuda() for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)).cuda() for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 256
        g = StyleGAN2Generator(**cfg_).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        # set mix_prob as 1 and 0 to cover all lines
        g.mix_prob = 1
        res = g(None, num_batches=2)
        g.mix_prob = 0
        res = g(None, num_batches=2)

        # test cond channels is negative number
        cfg_ = deepcopy(self.default_cfg)
        cfg_['cond_channels'] = -1
        g = StyleGAN2Generator(**cfg_)
        assert not hasattr(g, 'embed')

        # test cond channels > 0
        cfg_ = deepcopy(self.default_cfg)
        cfg_['cond_channels'] = 10
        g = StyleGAN2Generator(**cfg_)
        assert hasattr(g, 'embed')
        # test raise error
        with pytest.raises(AssertionError):
            g(None, num_batches=2)
        res = g(None, num_batches=2, cond=torch.randn(2, 10))
        assert res.shape == (2, 3, 64, 64)
