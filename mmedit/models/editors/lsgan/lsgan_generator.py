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
