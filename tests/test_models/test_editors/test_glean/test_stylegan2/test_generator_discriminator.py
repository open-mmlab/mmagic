# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from mmedit.models.editors.stylegan2 import (StyleGAN2Discriminator,
                                             StyleGAN2Generator)
from mmedit.models.utils import get_module_device


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


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
class TestStyleGANv2Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64, channel_multiplier=1)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_stylegan2_disc_cpu(self):
        d = StyleGAN2Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_stylegan2_disc_cuda(self):
        d = StyleGAN2Discriminator(**self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)


def test_get_module_device_cpu():
    device = get_module_device(nn.Conv2d(3, 3, 3, 1, 1))
    assert device == torch.device('cpu')

    # The input module should contain parameters.
    with pytest.raises(ValueError):
        get_module_device(nn.Flatten())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
def test_get_module_device_cuda():
    module = nn.Conv2d(3, 3, 3, 1, 1).cuda()
    device = get_module_device(module)
    assert device == next(module.parameters()).get_device()

    # The input module should contain parameters.
    with pytest.raises(ValueError):
        get_module_device(nn.Flatten().cuda())
