# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmagic.models.utils import generation_init_weights
from mmagic.registry import MODELS
from .cyclegan_modules import ResidualBlockWithDropout


@MODELS.register_module()
class ResnetGenerator(BaseModule):
    """Construct a Resnet-based generator that consists of residual blocks
    between a few downsampling/upsampling operations.

    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        base_channels (int): Number of filters at the last conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
        num_blocks (int): Number of residual blocks. Default: 9.
        padding_mode (str): The name of padding layer in conv layers:
            'reflect' | 'replicate' | 'zeros'. Default: 'reflect'.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=64,
                 norm_cfg=dict(type='IN'),
                 use_dropout=False,
                 num_blocks=9,
                 padding_mode='reflect',
                 init_cfg=dict(type='normal', gain=0.02)):
        super().__init__(init_cfg=init_cfg)
        assert num_blocks >= 0, ('Number of residual blocks must be '
                                 f'non-negative, but got {num_blocks}.')
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the resnet generator.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'

        model = []
        model += [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=7,
                padding=3,
                bias=use_bias,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode)
        ]

        num_down = 2
        # add downsampling layers
        for i in range(num_down):
            multiple = 2**i
            model += [
                ConvModule(
                    in_channels=base_channels * multiple,
                    out_channels=base_channels * multiple * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    norm_cfg=norm_cfg)
            ]

        # add residual blocks
        multiple = 2**num_down
        for i in range(num_blocks):
            model += [
                ResidualBlockWithDropout(
                    base_channels * multiple,
                    padding_mode=padding_mode,
                    norm_cfg=norm_cfg,
                    use_dropout=use_dropout)
            ]

        # add upsampling layers
        for i in range(num_down):
            multiple = 2**(num_down - i)
            model += [
                ConvModule(
                    in_channels=base_channels * multiple,
                    out_channels=base_channels * multiple // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    conv_cfg=dict(type='deconv', output_padding=1),
                    norm_cfg=norm_cfg)
            ]

        model += [
            ConvModule(
                in_channels=base_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='Tanh'),
                padding_mode=padding_mode)
        ]

        self.model = nn.Sequential(*model)
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def init_weights(self):
        """Initialize weights for the model."""
        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            super().init_weights()
            return
        generation_init_weights(
            self, init_type=self.init_type, init_gain=self.init_gain)
        self._is_init = True
