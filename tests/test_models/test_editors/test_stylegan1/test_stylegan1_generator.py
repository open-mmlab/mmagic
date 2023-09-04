# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch

from mmagic.models.editors.stylegan1 import StyleGAN1Generator
from mmagic.utils import register_all_modules

register_all_modules()


class TestStyleGAN1Generator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            out_size=256,
            style_channels=512,
            num_mlps=8,
            blur_kernel=[1, 2, 1],
            lr_mlp=0.01,
            default_style_mode='mix',
            eval_style_mode='single',
            mix_prob=0.9)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_g_cuda(self):
        # test default config
        g = StyleGAN1Generator(**self.default_cfg).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        styles = [torch.randn((1, 512)).cuda() for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)).cuda() for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 128
        g = StyleGAN1Generator(**cfg_).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 128, 128)

        # test generate function
        truncation_latent = g.get_mean_latent()
        assert truncation_latent.shape == (1, 512)
        style_mixing_images = g.style_mixing(
            curr_scale=32,
            truncation_latent=truncation_latent,
            n_source=4,
            n_target=4)
        assert style_mixing_images.shape == (25, 3, 32, 32)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_g_cpu(self):
        # test default config
        g = StyleGAN1Generator(**self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        styles = [torch.randn((1, 512)) for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)) for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 128
        g = StyleGAN1Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 128, 128)

        # test generate function
        truncation_latent = g.get_mean_latent()
        assert truncation_latent.shape == (1, 512)
        style_mixing_images = g.style_mixing(
            curr_scale=32,
            truncation_latent=truncation_latent,
            n_source=4,
            n_target=4)
        assert style_mixing_images.shape == (25, 3, 32, 32)

        # set mix_prob as 1.0 and 0.0 to force cover lines
        cfg_ = deepcopy(self.default_cfg)
        cfg_['mix_prob'] = 1
        g = StyleGAN1Generator(**cfg_)
        res = g(torch.randn, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['mix_prob'] = 1
        g = StyleGAN1Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['mix_prob'] = 0
        g = StyleGAN1Generator(**cfg_)
        res = g(torch.randn, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['mix_prob'] = 0
        g = StyleGAN1Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
