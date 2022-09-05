# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.registry import MODELS, MODULES
from ...utils import get_module_device


@MODULES.register_module()
class LSGANGenerator(nn.Module):
    """Generator for LSGAN.

    Implementation Details for LSGAN architecture:

    #. Adopt transposed convolution in the generator;
    #. Use batchnorm in the generator except for the final output layer;
    #. Use ReLU in the generator in addition to the final output layer;
    #. Keep channels of feature maps unchanged in the convolution backbone;
    #. Use one more 3x3 conv every upsampling in the convolution backbone.

    We follow the implementation details of the origin paper:
    Least Squares Generative Adversarial Networks
    https://arxiv.org/pdf/1611.04076.pdf

    Args:
        output_scale (int, optional): Output scale for the generated image.
            Defaults to 128.
        out_channels (int, optional): The channel number of the output feature.
            Defaults to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Defaults to 256.
        input_scale (int, optional): The scale of the input 2D feature map.
            Defaults to 8.
        noise_size (int, optional): Size of the input noise
            vector. Defaults to 1024.
        conv_cfg (dict, optional): Config for the convolution module used in
            this generator. Defaults to dict(type='ConvTranspose2d').
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to dict(type='BN').
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to dict(type='ReLU').
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to dict(type='Tanh').
    """

    def __init__(self,
                 output_scale=128,
                 out_channels=3,
                 base_channels=256,
                 input_scale=8,
                 noise_size=1024,
                 conv_cfg=dict(type='ConvTranspose2d'),
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='Tanh')):
        super().__init__()
        assert output_scale % input_scale == 0
        assert output_scale // input_scale >= 4

        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size

        self.noise2feat_head = nn.Sequential(
            nn.Linear(noise_size, input_scale * input_scale * base_channels))
        self.noise2feat_tail = nn.Sequential(nn.BatchNorm2d(base_channels))
        if default_act_cfg is not None:
            self.noise2feat_tail.add_module('act',
                                            MODELS.build(default_act_cfg))

        # the number of times for upsampling
        self.num_upsamples = int(np.log2(output_scale // input_scale)) - 2

        # build up convolution backbone (excluding the output layer)
        self.conv_blocks = nn.ModuleList()
        for _ in range(self.num_upsamples):
            self.conv_blocks.append(
                ConvModule(
                    base_channels,
                    base_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(conv_cfg, output_padding=1),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))
            self.conv_blocks.append(
                ConvModule(
                    base_channels,
                    base_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))

        # output blocks
        self.conv_blocks.append(
            ConvModule(
                base_channels,
                int(base_channels // 2),
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=dict(conv_cfg, output_padding=1),
                norm_cfg=default_norm_cfg,
                act_cfg=default_act_cfg))
        self.conv_blocks.append(
            ConvModule(
                int(base_channels // 2),
                int(base_channels // 4),
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=dict(conv_cfg, output_padding=1),
                norm_cfg=default_norm_cfg,
                act_cfg=default_act_cfg))
        self.conv_blocks.append(
            ConvModule(
                int(base_channels // 4),
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=out_act_cfg))

    def forward(self, noise, num_batches=0, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img`` and
                ``noise_batch`` will be returned.
        """
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            if noise.ndim == 2:
                noise_batch = noise
            else:
                raise ValueError('The noise should be in shape of (n, c)'
                                 f'but got {noise.shape}')
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.noise_size))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))
        # noise2feat
        x = self.noise2feat_head(noise_batch)
        x = x.reshape(
            (-1, self.base_channels, self.input_scale, self.input_scale))
        x = self.noise2feat_tail(x)
        # conv module
        for conv in self.conv_blocks:
            x = conv(x)

        if return_noise:
            return dict(fake_img=x, noise_batch=noise_batch)

        return x


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
