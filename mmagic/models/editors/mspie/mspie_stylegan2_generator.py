# Copyright (c) OpenMMLab. All rights reserved.
import random
from copy import deepcopy

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmagic.registry import MODELS
from ...utils import get_module_device
from ..pggan import PixelNorm
from ..stylegan1 import (ConstantInput, EqualLinearActModule, get_mean_latent,
                         style_mixing)
from ..stylegan2 import ModulatedToRGB
from .mspie_stylegan2_modules import ModulatedPEStyleConv


@MODELS.register_module()
class MSStyleGANv2Generator(BaseModule):
    """StyleGAN2 Generator.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of convolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    Args:
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
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
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 no_pad=False,
                 deconv2conv=False,
                 interp_pad=None,
                 up_config=dict(scale_factor=2, mode='nearest'),
                 up_after_conv=False,
                 head_pos_encoding=None,
                 head_pos_size=(4, 4),
                 interp_head=False):
        super().__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.num_mlps = num_mlps
        self.channel_multiplier = channel_multiplier
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob
        self.no_pad = no_pad
        self.deconv2conv = deconv2conv
        self.interp_pad = interp_pad
        self.with_interp_pad = interp_pad is not None
        self.up_config = deepcopy(up_config)
        self.up_after_conv = up_after_conv
        self.head_pos_encoding = head_pos_encoding
        self.head_pos_size = head_pos_size
        self.interp_head = interp_head

        # define style mapping layers
        mapping_layers = [PixelNorm()]

        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.),
                    act_cfg=dict(type='fused_bias')))

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        in_ch = self.channels[4]
        # constant input layer
        if self.head_pos_encoding:
            if self.head_pos_encoding['type'] in [
                    'CatersianGrid', 'CSG', 'CSG2d'
            ]:
                in_ch = 2
            self.head_pos_enc = MODELS.build(self.head_pos_encoding)
        else:
            size_ = 4
            if self.no_pad:
                size_ += 2
            self.constant_input = ConstantInput(self.channels[4], size=size_)

        # 4x4 stage
        self.conv1 = ModulatedPEStyleConv(
            in_ch,
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel,
            deconv2conv=self.deconv2conv,
            no_pad=self.no_pad,
            up_config=self.up_config,
            interp_pad=self.interp_pad)
        self.to_rgb1 = ModulatedToRGB(
            self.channels[4], style_channels, upsample=False)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channels_ = self.channels[2**i]

            self.convs.append(
                ModulatedPEStyleConv(
                    in_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    deconv2conv=self.deconv2conv,
                    no_pad=self.no_pad,
                    up_config=self.up_config,
                    interp_pad=self.interp_pad,
                    up_after_conv=self.up_after_conv))
            self.convs.append(
                ModulatedPEStyleConv(
                    out_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=False,
                    blur_kernel=blur_kernel,
                    deconv2conv=self.deconv2conv,
                    no_pad=self.no_pad,
                    up_config=self.up_config,
                    interp_pad=self.interp_pad,
                    up_after_conv=self.up_after_conv))
            self.to_rgbs.append(
                ModulatedToRGB(out_channels_, style_channels, upsample=True))

            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        noises = self.make_injected_noise()
        for layer_idx in range(self.num_injected_noises):
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 noises[layer_idx])

    def train(self, mode=True):
        """Set train/eval mode.

        Args:
            mode (bool, optional): Whether set train mode. Defaults to True.
        """
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

        return super(MSStyleGANv2Generator, self).train(mode)

    def make_injected_noise(self, chosen_scale=0):
        """make noises that will be injected into feature maps.

        Args:
            chosen_scale (int, optional): Chosen scale. Defaults to 0.

        Returns:
            list[Tensor]: List of layer-wise noise tensor.
        """
        device = get_module_device(self)

        base_scale = 2**2 + chosen_scale

        noises = [torch.randn(1, 1, base_scale, base_scale, device=device)]

        for i in range(3, self.log_size + 1):
            for n in range(2):
                _pad = 0
                if self.no_pad and not self.up_after_conv and n == 0:
                    _pad = 2
                noises.append(
                    torch.randn(
                        1,
                        1,
                        base_scale * 2**(i - 2) + _pad,
                        base_scale * 2**(i - 2) + _pad,
                        device=device))

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
                     chosen_scale=0):
        """Generating style mixing images.

        Args:
            n_source (int): Number of source images.
            n_target (int): Number of target images.
            inject_index (int, optional): Index from which replace with source
                latent. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            curr_scale (int): Current image scale. Defaults to -1.
            transition_weight (float, optional): The weight used in resolution
                transition. Defaults to 1.0.
            chosen_scale (int, optional): Chosen scale. Defaults to 0.
        Returns:
            torch.Tensor: Table of style-mixing images.
        """
        return style_mixing(
            self,
            n_source=n_source,
            n_target=n_target,
            inject_index=inject_index,
            truncation_latent=truncation_latent,
            truncation=truncation,
            style_channels=self.style_channels,
            chosen_scale=chosen_scale)

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
                chosen_scale=0):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
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
            elif chosen_scale > 0:
                if not hasattr(self, f'injected_noise_{chosen_scale}_0'):
                    noises_ = self.make_injected_noise(chosen_scale)
                    for i in range(self.num_injected_noises):
                        setattr(self, f'injected_noise_{chosen_scale}_{i}',
                                noises_[i])
                injected_noise = [
                    getattr(self, f'injected_noise_{chosen_scale}_{i}')
                    for i in range(self.num_injected_noises)
                ]
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

        if isinstance(chosen_scale, int):
            chosen_scale = (chosen_scale, chosen_scale)

        # 4x4 stage
        if self.head_pos_encoding:
            if self.interp_head:
                out = self.head_pos_enc.make_grid2d(self.head_pos_size[0],
                                                    self.head_pos_size[1],
                                                    latent.size(0))
                h_in = self.head_pos_size[0] + chosen_scale[0]
                w_in = self.head_pos_size[1] + chosen_scale[1]
                out = F.interpolate(
                    out,
                    size=(h_in, w_in),
                    mode='bilinear',
                    align_corners=True)
            else:
                out = self.head_pos_enc.make_grid2d(
                    self.head_pos_size[0] + chosen_scale[0],
                    self.head_pos_size[1] + chosen_scale[1], latent.size(0))
            out = out.to(latent)
        else:
            out = self.constant_input(latent)
            if chosen_scale[0] != 0 or chosen_scale[1] != 0:
                out = F.interpolate(
                    out,
                    size=(out.shape[2] + chosen_scale[0],
                          out.shape[3] + chosen_scale[1]),
                    mode='bilinear',
                    align_corners=True)

        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbs):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)

            _index += 2

        img = skip

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch,
                injected_noise=injected_noise)
            return output_dict

        return img
