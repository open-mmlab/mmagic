# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.runner import load_state_dict

from mmedit.registry import MODULES
from .singan_modules import GeneratorBlock


@MODULES.register_module()
class SinGANMultiScaleGenerator(nn.Module):
    """Multi-Scale Generator used in SinGAN.

    More details can be found in: Singan: Learning a Generative Model from a
    Single Natural Image, ICCV'19.

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
                 **kwargs):
        super().__init__()

        self.pad_head = int((kernel_size - 1) / 2 * num_layers)
        self.blocks = nn.ModuleList()

        self.upsample = partial(
            F.interpolate, mode='bicubic', align_corners=True)

        for scale in range(num_scales + 1):
            base_ch = min(base_channels * pow(2, int(np.floor(scale / 4))),
                          128)
            min_feat_ch = min(
                min_feat_channels * pow(2, int(np.floor(scale / 4))), 128)

            self.blocks.append(
                GeneratorBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_layers=num_layers,
                    base_channels=base_ch,
                    min_feat_channels=min_feat_ch,
                    out_act_cfg=out_act_cfg,
                    **kwargs))

        self.noise_padding_layer = nn.ZeroPad2d(self.pad_head)
        self.img_padding_layer = nn.ZeroPad2d(self.pad_head)

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
            input_sample = torch.zeros(
                (num_batches, 3, fixed_noises[0].shape[-2],
                 fixed_noises[0].shape[-1])).to(fixed_noises[0])

        g_res = input_sample

        for stage in range(curr_scale + 1):
            if rand_mode == 'recon':
                noise_ = fixed_noises[stage]
            else:
                noise_ = torch.randn(num_batches,
                                     *fixed_noises[stage].shape[1:]).to(g_res)
            if return_noise:
                noise_list.append(noise_)

            # add padding at head
            pad_ = (self.pad_head, ) * 4
            noise_ = F.pad(noise_, pad_)
            g_res_pad = F.pad(g_res, pad_)
            noise = noise_ * noise_weights[stage] + g_res_pad

            g_res = self.blocks[stage](noise.detach(), g_res)

            if get_prev_res and stage != curr_scale:
                prev_res_list.append(g_res)

            # upsample, here we use interpolation from PyTorch
            if stage != curr_scale:
                h_next, w_next = fixed_noises[stage + 1].shape[-2:]
                g_res = self.upsample(g_res, (h_next, w_next))

        if get_prev_res or return_noise:
            output_dict = dict(
                fake_img=g_res,
                prev_res_list=prev_res_list,
                # noise_batch=noise_list
            )
            return output_dict

        return g_res

    def check_and_load_prev_weight(self, curr_scale):
        logger = MMLogger.get_current_instance()
        if curr_scale == 0:
            return
        prev_ch = self.blocks[curr_scale - 1].base_channels
        curr_ch = self.blocks[curr_scale].base_channels

        prev_in_ch = self.blocks[curr_scale - 1].in_channels
        curr_in_ch = self.blocks[curr_scale].in_channels
        if prev_ch == curr_ch and prev_in_ch == curr_in_ch:
            load_state_dict(
                self.blocks[curr_scale],
                self.blocks[curr_scale - 1].state_dict(),
                logger=MMLogger.get_current_instance())
            logger.info('Successfully load pretrianed model from last scale.')
        else:
            logger.info(
                'Cannot load pretrained model from last scale since'
                f' prev_ch({prev_ch}) != curr_ch({curr_ch})'
                f' or prev_in_ch({prev_in_ch}) != curr_in_ch({curr_in_ch})')
