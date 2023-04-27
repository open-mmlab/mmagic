# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model import constant_init, normal_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm


class GeneratorBlock(nn.Module):
    """Generator block used in SinGAN.

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
        stride (int, optional): Same as :obj:`nn.Conv2d`. Defaults to 1.
        allow_no_residual (bool, optional): Whether to allow no residual link
            in this block. Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 num_layers,
                 base_channels,
                 min_feat_channels,
                 out_act_cfg=dict(type='Tanh'),
                 stride=1,
                 allow_no_residual=False,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels

        self.base_channels = base_channels

        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.allow_no_residual = allow_no_residual

        self.head = ConvModule(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            **kwargs)

        self.body = nn.Sequential()

        for i in range(num_layers - 2):
            feat_channels_ = int(base_channels / pow(2, (i + 1)))
            block = ConvModule(
                max(2 * feat_channels_, min_feat_channels),
                max(feat_channels_, min_feat_channels),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                **kwargs)
            self.body.add_module(f'block{i+1}', block)

        self.tail = ConvModule(
            max(feat_channels_, min_feat_channels),
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            norm_cfg=None,
            act_cfg=out_act_cfg,
            **kwargs)

        self.init_weights()

    def forward(self, x, prev):
        """Forward function.

        Args:
            x (Tensor): Input feature map.
            prev (Tensor): Previous feature map.

        Returns:
            Tensor: Output feature map with the shape of (N, C, H, W).
        """
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        # if prev and x are not in the same shape at the channel dimension
        if self.allow_no_residual and x.shape[1] != prev.shape[1]:
            return x

        return x + prev

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, (_BatchNorm, nn.InstanceNorm2d)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')


class DiscriminatorBlock(nn.Module):
    """Discriminator Block used in SinGAN.

    Args:
        in_channels (int): Input channels.
        base_channels (int): Base channels for this block.
        min_feat_channels (int): The minimum channels for feature map.
        kernel_size (int): Size of convolutional kernel, same as
            :obj:`nn.Conv2d`.
        padding (int): Padding for convolutional layer, same as
            :obj:`nn.Conv2d`.
        num_layers (int): The number of convolutional layers in this block.
        norm_cfg (dict | None, optional): Config for the normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict | None, optional): Config for the activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.2).
        stride (int, optional): The stride for the convolutional layer, same as
            :obj:`nn.Conv2d`. Defaults to 1.
    """

    def __init__(self,
                 in_channels,
                 base_channels,
                 min_feat_channels,
                 kernel_size,
                 padding,
                 num_layers,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 stride=1,
                 **kwargs):
        super().__init__()

        self.base_channels = base_channels
        self.stride = stride
        self.head = ConvModule(
            in_channels,
            base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.body = nn.Sequential()

        for i in range(num_layers - 2):
            feat_channels_ = int(base_channels / pow(2, (i + 1)))
            block = ConvModule(
                max(2 * feat_channels_, min_feat_channels),
                max(feat_channels_, min_feat_channels),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs)
            self.body.add_module(f'block{i+1}', block)

        self.tail = ConvModule(
            max(feat_channels_, min_feat_channels),
            1,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            norm_cfg=None,
            act_cfg=None,
            **kwargs)

        self.init_weights()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x

    # TODO: study the effects of init functions
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, (_BatchNorm, nn.InstanceNorm2d)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
