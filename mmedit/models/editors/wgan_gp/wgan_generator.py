# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.utils import get_module_device
from mmedit.registry import MODELS, MODULES
from .wgan_gp_module import WGANNoiseTo2DFeat


@MODULES.register_module()
class WGANGPGenerator(nn.Module):
    r"""Generator for WGANGP.

    Implementation Details for WGANGP generator the same as training
    configuration (a) described in PGGAN paper:
    PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION
    https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf # noqa

    #. Adopt convolution architecture specified in appendix A.2;
    #. Use batchnorm in the generator except for the final output layer;
    #. Use ReLU in the generator except for the final output layer;
    #. Use Tanh in the last layer;
    #. Initialize all weights using He’s initializer.

    Args:
        noise_size (int): Size of the input noise vector.
        out_scale (int): Output scale for the generated image.
        conv_module_cfg (dict, optional): Config for the convolution
            module used in this generator. Defaults to None.
        upsample_cfg (dict, optional): Config for the upsampling operation.
            Defaults to None.
    """
    _default_channels_per_scale = {
        '4': 512,
        '8': 512,
        '16': 256,
        '32': 128,
        '64': 64,
        '128': 32
    }
    _default_conv_module_cfg = dict(
        conv_cfg=None,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        act_cfg=dict(type='ReLU'),
        norm_cfg=dict(type='BN'),
        order=('conv', 'norm', 'act'))

    _default_upsample_cfg = dict(type='nearest', scale_factor=2)

    def __init__(self,
                 noise_size,
                 out_scale,
                 conv_module_cfg=None,
                 upsample_cfg=None):
        super().__init__()
        # set initial params
        self.noise_size = noise_size
        self.out_scale = out_scale
        self.conv_module_cfg = deepcopy(self._default_conv_module_cfg)
        if conv_module_cfg is not None:
            self.conv_module_cfg.update(conv_module_cfg)
        self.upsample_cfg = upsample_cfg if upsample_cfg else deepcopy(
            self._default_upsample_cfg)
        # set noise2feat head
        self.noise2feat = WGANNoiseTo2DFeat(
            self.noise_size, self._default_channels_per_scale['4'])
        # set conv_blocks
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ConvModule(512, 512, **self.conv_module_cfg))

        log2scale = int(np.log2(self.out_scale))
        for i in range(3, log2scale + 1):
            self.conv_blocks.append(MODELS.build(self._default_upsample_cfg))
            self.conv_blocks.append(
                ConvModule(self._default_channels_per_scale[str(2**(i - 1))],
                           self._default_channels_per_scale[str(2**i)],
                           **self.conv_module_cfg))
            self.conv_blocks.append(
                ConvModule(self._default_channels_per_scale[str(2**i)],
                           self._default_channels_per_scale[str(2**i)],
                           **self.conv_module_cfg))
        self.to_rgb = ConvModule(
            self._default_channels_per_scale[str(self.out_scale)],
            kernel_size=1,
            out_channels=3,
            act_cfg=dict(type='Tanh'))

    def forward(self, noise, num_batches=0, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size. Defaults to
                0.
            return_noise (bool, optional):  If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img`` and
                ``noise_batch`` will be returned.
        """
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            assert noise.ndim == 2, ('The noise should be in shape of (n, c), '
                                     f'but got {noise.shape}')
            noise_batch = noise
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

        # noise vector to 2D feature
        x = self.noise2feat(noise_batch)
        for conv in self.conv_blocks:
            x = conv(x)
        out_img = self.to_rgb(x)

        if return_noise:
            output = dict(fake_img=out_img, noise_batch=noise_batch)
            return output

        return out_img
