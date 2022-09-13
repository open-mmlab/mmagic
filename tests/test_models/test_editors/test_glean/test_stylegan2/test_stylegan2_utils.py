# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmedit.models.editors.stylegan1.stylegan_utils import (get_mean_latent,
                                                            style_mixing)
from mmedit.models.editors.stylegan2 import StyleGAN2Generator
from mmedit.models.utils import get_module_device


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
