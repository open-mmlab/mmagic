# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmedit.registry import MODULES
from ..singan.singan_generator import SinGANMultiScaleGenerator
from ..singan.singan_modules import GeneratorBlock


@MODULES.register_module()
class SinGANMSGeneratorPE(SinGANMultiScaleGenerator):
    """Multi-Scale Generator used in SinGAN with positional encoding.

    More details can be found in: Positional Encoding as Spatial Inductvie Bias
    in GANs, CVPR'2021.

    Notes:

    - In this version, we adopt the interpolation function from the official
      PyTorch APIs, which is different from the original implementation by the
      authors. However, in our experiments, this influence can be ignored.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_scales (int): The number of scales/stages in generator. Note
            that this number is counted from zero, which is the same as the
            original paper.
        kernel_size (int, optional): Kernel size, same as :obj:`nn.Conv2d`.
            Defaults to 3.
        padding (int, optional): Padding for the convolutional layer, same as
            :obj:`nn.Conv2d`. Defaults to 0.
        num_layers (int, optional): The number of convolutional layers in each
            generator block. Defaults to 5.
        base_channels (int, optional): The basic channels for convolutional
            layers in the generator block. Defaults to 32.
        min_feat_channels (int, optional): Minimum channels for the feature
            maps in the generator block. Defaults to 32.
        out_act_cfg (dict | None, optional): Configs for output activation
            layer. Defaults to dict(type='Tanh').
        padding_mode (str, optional): The mode of convolutional padding, same
            as :obj:`nn.Conv2d`. Defaults to 'zero'.
        pad_at_head (bool, optional): Whether to add padding at head.
            Defaults to True.
        interp_pad (bool, optional): The padding value of interpolating feature
            maps. Defaults to False.
        noise_with_pad (bool, optional): Whether the input fixed noises are
            with explicit padding. Defaults to False.
        positional_encoding (dict | None, optional): Configs for the positional
            encoding. Defaults to None.
        first_stage_in_channels (int | None, optional): The input channel of
            the first generator block. If None, the first stage will adopt the
            same input channels as other stages. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_scales,
                 kernel_size=3,
                 padding=0,
                 num_layers=5,
                 base_channels=32,
                 min_feat_channels=32,
                 out_act_cfg=dict(type='Tanh'),
                 padding_mode='zero',
                 pad_at_head=True,
                 interp_pad=False,
                 noise_with_pad=False,
                 positional_encoding=None,
                 first_stage_in_channels=None,
                 **kwargs):
        super(SinGANMultiScaleGenerator, self).__init__()

        self.pad_at_head = pad_at_head
        self.interp_pad = interp_pad
        self.noise_with_pad = noise_with_pad

        self.with_positional_encode = positional_encoding is not None
        if self.with_positional_encode:
            self.head_position_encode = MODULES.build(positional_encoding)

        self.pad_head = int((kernel_size - 1) / 2 * num_layers)
        self.blocks = nn.ModuleList()

        self.upsample = partial(
            F.interpolate, mode='bicubic', align_corners=True)

        for scale in range(num_scales + 1):
            base_ch = min(base_channels * pow(2, int(np.floor(scale / 4))),
                          128)
            min_feat_ch = min(
                min_feat_channels * pow(2, int(np.floor(scale / 4))), 128)

            if scale == 0:
                in_ch = (
                    first_stage_in_channels
                    if first_stage_in_channels else in_channels)
            else:
                in_ch = in_channels

            self.blocks.append(
                GeneratorBlock(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_layers=num_layers,
                    base_channels=base_ch,
                    min_feat_channels=min_feat_ch,
                    out_act_cfg=out_act_cfg,
                    padding_mode=padding_mode,
                    **kwargs))

        if padding_mode == 'zero':
            self.noise_padding_layer = nn.ZeroPad2d(self.pad_head)
            self.img_padding_layer = nn.ZeroPad2d(self.pad_head)
            self.mask_padding_layer = nn.ReflectionPad2d(self.pad_head)
        elif padding_mode == 'reflect':
            self.noise_padding_layer = nn.ReflectionPad2d(self.pad_head)
            self.img_padding_layer = nn.ReflectionPad2d(self.pad_head)
            self.mask_padding_layer = nn.ReflectionPad2d(self.pad_head)
            mmengine.print_log('Using Reflection padding', 'current')
        else:
            raise NotImplementedError(
                f'Padding mode {padding_mode} is not supported')

    def forward(self,
                input_sample,
                fixed_noises,
                noise_weights,
                rand_mode,
                curr_scale,
                num_batches=1,
                get_prev_res=False,
                return_noise=False):
        """Forward function.

        Args:
            input_sample (Tensor | None): The input for generator. In the
                original implementation, a tensor filled with zeros is adopted.
                If None is given, we will construct it from the first fixed
                noises.
            fixed_noises (list[Tensor]): List of the fixed noises in SinGAN.
            noise_weights (list[float]): List of the weights for random noises.
            rand_mode (str): Choices from ['rand', 'recon']. In ``rand`` mode,
                it will sample from random noises. Otherwise, the
                reconstruction for the single image will be returned.
            curr_scale (int): The scale for the current inference or training.
            num_batches (int, optional): The number of batches. Defaults to 1.
            get_prev_res (bool, optional): Whether to return results from
                previous stages. Defaults to False.
            return_noise (bool, optional): Whether to return noises tensor.
                Defaults to False.

        Returns:
            Tensor | dict: Generated image tensor or dictionary containing \
                more data.
        """
        if get_prev_res or return_noise:
            prev_res_list = []
            noise_list = []

        if input_sample is None:
            h, w = fixed_noises[0].shape[-2:]
            if self.noise_with_pad:
                h -= 2 * self.pad_head
                w -= 2 * self.pad_head
            input_sample = torch.zeros(
                (num_batches, 3, h, w)).to(fixed_noises[0])

        g_res = input_sample

        for stage in range(curr_scale + 1):
            if rand_mode == 'recon':
                noise_ = fixed_noises[stage]
            else:
                noise_ = torch.randn(num_batches,
                                     *fixed_noises[stage].shape[1:]).to(g_res)
            if return_noise:
                noise_list.append(noise_)

            if self.with_positional_encode and stage == 0:
                head_grid = self.head_position_encode(fixed_noises[0])
                noise_ = noise_ + head_grid

            # add padding at head
            if self.pad_at_head:
                if self.interp_pad:
                    if self.noise_with_pad:
                        size = noise_.shape[-2:]
                    else:
                        size = (noise_.size(2) + 2 * self.pad_head,
                                noise_.size(3) + 2 * self.pad_head)
                        noise_ = self.upsample(noise_, size)
                    g_res_pad = self.upsample(g_res, size)
                else:
                    if not self.noise_with_pad:
                        noise_ = self.noise_padding_layer(noise_)
                    g_res_pad = self.img_padding_layer(g_res)
            else:
                g_res_pad = g_res

            if stage == 0 and self.with_positional_encode:
                noise = noise_ * noise_weights[stage]
            else:
                noise = noise_ * noise_weights[stage] + g_res_pad
            g_res = self.blocks[stage](noise.detach(), g_res)

            if get_prev_res and stage != curr_scale:
                prev_res_list.append(g_res)

            # upsample, here we use interpolation from PyTorch
            if stage != curr_scale:
                h_next, w_next = fixed_noises[stage + 1].shape[-2:]
                if self.noise_with_pad:
                    # remove the additional padding if noise with pad
                    h_next -= 2 * self.pad_head
                    w_next -= 2 * self.pad_head
                g_res = self.upsample(g_res, (h_next, w_next))

        if get_prev_res or return_noise:
            output_dict = dict(
                fake_img=g_res,
                prev_res_list=prev_res_list,
                noise_batch=noise_list)
            return output_dict

        return g_res
