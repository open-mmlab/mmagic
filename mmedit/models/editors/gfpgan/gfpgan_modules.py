# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
from torch import nn
from torch.nn import functional as F

from .stylegan2_clean import StyleGAN2GeneratorClean


class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    It is the clean version without custom compiled CUDA extensions
        used in StyleGAN2.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels.
            Default: False.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 sft_half=False):
        super(StyleGAN2GeneratorCSFT, self).__init__(
            out_size,
            style_channels=num_style_feat,
            num_mlps=num_mlp,
            channel_multiplier=channel_multiplier)
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorCSFT.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor.
                Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
            Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mapping(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [
                    None
                ] * self.num_injected_noises  # for each style conv layer
            else:  # use the stored noise
                noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation *
                                        (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2],
                self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(
                        out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2],
                          skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ResBlock(nn.Module):
    """Residual block with bilinear upsampling/downsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        mode (str): Upsampling/downsampling mode.
        Options: down | up. Default: down.
    """

    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        # upsample/downsample
        out = F.interpolate(
            out,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        # skip
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)
        skip = self.skip(x)
        out = out + skip
        return out
