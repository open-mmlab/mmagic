# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmengine import print_log

from mmedit.registry import MODULES
from .singan_modules import DiscriminatorBlock


@MODULES.register_module()
class SinGANMultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator used in SinGAN.

    More details can be found in: Singan: Learning a Generative Model from a
    Single Natural Image, ICCV'19.

    Args:
        in_channels (int): Input channels.
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
    """

    def __init__(self,
                 in_channels,
                 num_scales,
                 kernel_size=3,
                 padding=0,
                 num_layers=5,
                 base_channels=32,
                 min_feat_channels=32,
                 **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        for scale in range(num_scales + 1):
            base_ch = min(base_channels * pow(2, int(np.floor(scale / 4))),
                          128)
            min_feat_ch = min(
                min_feat_channels * pow(2, int(np.floor(scale / 4))), 128)
            self.blocks.append(
                DiscriminatorBlock(
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_layers=num_layers,
                    base_channels=base_ch,
                    min_feat_channels=min_feat_ch,
                    **kwargs))

    def forward(self, x, curr_scale):
        """Forward function.

        Args:
            x (Tensor): Input feature map.
            curr_scale (int): Current scale for discriminator. If in testing,
                you need to set it to the last scale.

        Returns:
            Tensor: Discriminative results.
        """
        out = self.blocks[curr_scale](x)
        return out

    def check_and_load_prev_weight(self, curr_scale):
        if curr_scale == 0:
            return
        prev_ch = self.blocks[curr_scale - 1].base_channels
        curr_ch = self.blocks[curr_scale].base_channels
        if prev_ch == curr_ch:
            self.blocks[curr_scale].load_state_dict(
                self.blocks[curr_scale - 1].state_dict())
            print_log('Successfully load pretrianed model from last scale.')
        else:
            print_log('Cannot load pretrained model from last scale since'
                      f' prev_ch({prev_ch}) != curr_ch({curr_ch})')
