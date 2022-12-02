# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn

from mmedit.registry import MODELS

# Note: This model is copied from Disco-Diffusion colab.
# SourceCode: https://colab.research.google.com/drive/1uGKaBOEACeinAA7jX1_zSFtj_ZW-huHS#scrollTo=XIqUfrmvLIhg # noqa


def append_dims(x, n):
    """Append dims."""
    return x[(Ellipsis, *(None, ) * (n - x.ndim))]


def expand_to_planes(x, shape):
    """Expand tensor to planes."""
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    """convert alpha&sigma to timestep."""
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    """convert timestep to alpha and sigma."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


class ConvBlock(nn.Sequential):
    """Convolution Block.

    Args:
        c_in (int): Input channels.
        c_out (int): Output channels.
    """

    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    """Skip block wrapper. Wrapping main block and skip block and concat their
    outputs together.

    Args:
        main (list): A list of main modules.
        skip (nn.Module): Skip Module. If not given,
            set to ``nn.Identity()``. Defaults to None.
    """

    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        """Forward function."""
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    """Fourier features mapping MLP.

    Args:
        in_features (int): Input channels.
        out_features (int): Output channels.
        std (float): Standard deviation. Defaults to 1..
    """

    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(
            torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        """Forward function."""
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


@MODELS.register_module()
class SecondaryDiffusionImageNet2(nn.Module):
    """A smaller secondary diffusion model trained by Katherine Crowson to
    remove noise from intermediate timesteps to prepare them for CLIP.

    Ref: https://twitter.com/rivershavewings/status/1462859669454536711 # noqa
    """

    def __init__(self):
        super().__init__()
        self.in_channels = 3
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        """Forward function."""
        timestep_embed = expand_to_planes(
            self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(
            partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return dict(v=v, pred=pred, eps=eps)
