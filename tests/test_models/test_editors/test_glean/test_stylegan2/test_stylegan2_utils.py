# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.glean.stylegan2 import (StyleGANv2Generator,
                                                   stylegan2_utils)


def test_get_module_device():
    config = dict(
        out_size=64, style_channels=16, num_mlps=4, channel_multiplier=1)
    g = StyleGANv2Generator(**config)
    res = g(None, num_batches=2)
    assert res.shape == (2, 3, 64, 64)

    truncation_mean = stylegan2_utils.get_mean_latent(g, 4096)
    res = g(
        None,
        num_batches=2,
        randomize_noise=False,
        truncation=0.7,
        truncation_latent=truncation_mean)

    # res = g.style_mixing(2, 2, truncation_latent=truncation_mean)
    res = stylegan2_utils.style_mixing(
        g,
        n_source=2,
        n_target=2,
        truncation_latent=truncation_mean,
        style_channels=g.style_channels)

    assert stylegan2_utils.get_module_device(g) == torch.device('cpu')
