# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine import MMLogger
from mmengine.model import normal_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmedit.registry import MODULES
from ...utils import get_module_device


@MODULES.register_module()
class DCGANGenerator(nn.Module):
    """Generator for DCGAN.

    Implementation Details for DCGAN architecture:

    #. Adopt transposed convolution in the generator;
    #. Use batchnorm in the generator except for the final output layer;
    #. Use ReLU in the generator in addition to the final output layer.

    More details can be found in the original paper:
    Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks
    http://arxiv.org/abs/1511.06434

    Args:
        output_scale (int | tuple[int]): Output scale for the generated
            image. If only a integer is provided, the output image will
            be a square shape. The tuple of two integers will set the
            height and width for the output image, respectively.
        out_channels (int, optional): The channel number of the output feature.
            Default to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Default to 1024.
        input_scale (int | tuple[int], optional): Output scale for the
            generated image. If only a integer is provided, the input feature
            ahead of the convolutional generator will be a square shape. The
            tuple of two integers will set the height and width for the input
            convolutional feature, respectively. Defaults to 4.
        noise_size (int, optional): Size of the input noise
            vector. Defaults to 100.
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='ReLU')``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
        pretrained (str, optional): Path for the pretrained model. Default to
            ``None``.
    """

    def __init__(self,
                 output_scale,
                 out_channels=3,
                 base_channels=1024,
                 input_scale=4,
                 noise_size=100,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='Tanh'),
                 pretrained=None):
        super().__init__()
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size

        # the number of times for upsampling
        self.num_upsamples = int(np.log2(output_scale // input_scale))

        # output 4x4 feature map
        self.noise2feat = ConvModule(
            noise_size,
            base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg)

        # build up upsampling backbone (excluding the output layer)
        upsampling = []
        curr_channel = base_channels
        for _ in range(self.num_upsamples - 1):
            upsampling.append(
                ConvModule(
                    curr_channel,
                    curr_channel // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))

            curr_channel //= 2

        self.upsampling = nn.Sequential(*upsampling)

        # output layer
        self.output_layer = ConvModule(
            curr_channel,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=None,
            act_cfg=out_act_cfg)

        self.init_weights(pretrained=pretrained)

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
                noise_batch = noise[:, :, None, None]
            elif noise.ndim == 4:
                noise_batch = noise
            else:
                raise ValueError('The noise should be in shape of (n, c) or '
                                 f'(n, c, 1, 1), but got {noise.shape}')
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size, 1, 1))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.noise_size, 1, 1))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))

        x = self.noise2feat(noise_batch)
        x = self.upsampling(x)
        x = self.output_layer(x)

        if return_noise:
            return dict(fake_img=x, noise_batch=noise_batch)

        return x

    def init_weights(self, pretrained=None):
        """Init weights for models.

        We just use the initialization method proposed in the original paper.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, _BatchNorm):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
