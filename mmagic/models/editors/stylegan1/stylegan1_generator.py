# Copyright (c) OpenMMLab. All rights reserved.
import random

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmagic.registry import MODELS
from ...utils import get_module_device
from ..pggan import EqualizedLRConvModule, PixelNorm
from .stylegan1_modules import EqualLinearActModule, StyleConv
from .stylegan_utils import get_mean_latent, style_mixing


@MODELS.register_module('StyleGANv1Generator')
@MODELS.register_module()
class StyleGAN1Generator(BaseModule):
    """StyleGAN1 Generator.

    In StyleGAN1, we use a progressive growing architecture composing of a
    style mapping module and number of convolutional style blocks. More details
    can be found in: A Style-Based Generator Architecture for Generative
    Adversarial Networks CVPR2019.

    Args:
        out_size (int): The output size of the StyleGAN1 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 2, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we adopt mixing style mode in default. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to 0.9.
    """

    def __init__(self,
                 out_size,
                 style_channels,
                 num_mlps=8,
                 blur_kernel=[1, 2, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9):
        super().__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.num_mlps = num_mlps
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob

        # define style mapping layers
        mapping_layers = [PixelNorm()]

        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.),
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2)))

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(2, self.log_size + 1):
            out_channels_ = self.channels[2**i]
            self.convs.append(
                StyleConv(
                    in_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    initial=(i == 2),
                    upsample=True,
                    fused=True))
            self.to_rgbs.append(
                EqualizedLRConvModule(out_channels_, 3, 1, act_cfg=None))

            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents

        # register buffer for injected noises
        for layer_idx in range(self.num_injected_noises):
            res = (layer_idx + 4) // 2
            shape = [1, 1, 2**res, 2**res]
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 torch.randn(*shape))

    def train(self, mode=True):
        if mode:
            if self.default_style_mode != self._default_style_mode:
                mmengine.print_log(
                    f'Switch to train style mode: {self._default_style_mode}')
            self.default_style_mode = self._default_style_mode

        else:
            if self.default_style_mode != self.eval_style_mode:
                mmengine.print_log(
                    f'Switch to evaluation style mode: {self.eval_style_mode}')
            self.default_style_mode = self.eval_style_mode

        return super(StyleGAN1Generator, self).train(mode)

    def make_injected_noise(self):
        """make noises that will be injected into feature maps.

        Returns:
            list[Tensor]: List of layer-wise noise tensor.
        """
        device = get_module_device(self)

        # noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]
        noises = []

        for i in range(2, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        return get_mean_latent(self, num_samples, **kwargs)

    def style_mixing(self,
                     n_source,
                     n_target,
                     inject_index=1,
                     truncation_latent=None,
                     truncation=0.7,
                     curr_scale=-1,
                     transition_weight=1):
        return style_mixing(
            self,
            n_source=n_source,
            n_target=n_target,
            inject_index=inject_index,
            truncation=truncation,
            truncation_latent=truncation_latent,
            style_channels=self.style_channels,
            curr_scale=curr_scale,
            transition_weight=transition_weight)

    def forward(self,
                styles,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                injected_noise=None,
                randomize_noise=True,
                transition_weight=1.,
                curr_scale=-1):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN1, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.
            transition_weight (float, optional): The weight used in resolution
                transition. Defaults to 1..
            curr_scale (int, optional): The resolution scale of generated image
                tensor. -1 means the max resolution scale of the StyleGAN1.
                Defaults to -1.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """
        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmengine.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    noise_generator((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    torch.randn((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]

        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []
            # calculate truncation latent on the fly
            if truncation_latent is None and not hasattr(
                    self, 'truncation_latent'):
                self.truncation_latent = self.get_mean_latent()
                truncation_latent = self.truncation_latent
            elif truncation_latent is None and hasattr(self,
                                                       'truncation_latent'):
                truncation_latent = self.truncation_latent

            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        curr_log_size = self.log_size if curr_scale < 0 else int(
            np.log2(curr_scale))
        step = curr_log_size - 2

        _index = 0
        out = latent
        # 4x4 ---> higher resolutions
        for i, (conv, to_rgb) in enumerate(zip(self.convs, self.to_rgbs)):
            if i > 0 and step > 0:
                out_prev = out
            out = conv(
                out,
                latent[:, _index],
                latent[:, _index + 1],
                noise1=injected_noise[2 * i],
                noise2=injected_noise[2 * i + 1])
            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= transition_weight < 1:
                    skip_rgb = self.to_rgbs[i - 1](out_prev)
                    skip_rgb = F.interpolate(
                        skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - transition_weight
                           ) * skip_rgb + transition_weight * out
                break

            _index += 2

        img = out

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch)
            return output_dict

        return img
