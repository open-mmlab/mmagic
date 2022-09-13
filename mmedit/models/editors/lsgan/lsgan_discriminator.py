# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.registry import MODELS, MODULES


@MODULES.register_module()
class LSGANDiscriminator(nn.Module):
    """Discriminator for LSGAN.

    Implementation Details for LSGAN architecture:

    #. Adopt convolution in the discriminator;
    #. Use batchnorm in the discriminator except for the input and final \
       output layer;
    #. Use LeakyReLU in the discriminator in addition to the output layer;
    #. Use fully connected layer in the output layer;
    #. Use 5x5 conv rather than 4x4 conv in DCGAN.

    Args:
        input_scale (int, optional): The scale of the input image. Defaults to
            128.
        output_scale (int, optional): The final scale of the convolutional
            feature. Defaults to 8.
        out_channels (int, optional): The channel number of the final output
            layer. Defaults to 1.
        in_channels (int, optional): The channel number of the input image.
            Defaults to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Defaults to 128.
        conv_cfg (dict, optional): Config for the convolution module used in
            this discriminator. Defaults to dict(type='Conv2d').
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='LeakyReLU', negative_slope=0.2)``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
    """

    def __init__(self,
                 input_scale=128,
                 output_scale=8,
                 out_channels=1,
                 in_channels=3,
                 base_channels=64,
                 conv_cfg=dict(type='Conv2d'),
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 out_act_cfg=None):
        super().__init__()
        assert input_scale % output_scale == 0
        assert input_scale // output_scale >= 2

        self.input_scale = input_scale
        self.output_scale = output_scale
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.with_out_activation = out_act_cfg is not None

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            ConvModule(
                in_channels,
                base_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=default_act_cfg))

        # the number of times for downsampling
        self.num_downsamples = int(np.log2(input_scale // output_scale)) - 1

        # build up downsampling backbone (excluding the output layer)
        curr_channels = base_channels
        for _ in range(self.num_downsamples):
            self.conv_blocks.append(
                ConvModule(
                    curr_channels,
                    curr_channels * 2,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    conv_cfg=conv_cfg,
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))
            curr_channels = curr_channels * 2

        # output layer
        self.decision = nn.Sequential(
            nn.Linear(output_scale * output_scale * curr_channels,
                      out_channels))
        if self.with_out_activation:
            self.out_activation = MODELS.build(out_act_cfg)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Fake or real image tensor.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """
        n = x.shape[0]

        for conv in self.conv_blocks:
            x = conv(x)

        x = x.reshape(n, -1)
        x = self.decision(x)

        if self.with_out_activation:
            x = self.out_activation(x)

        return x
