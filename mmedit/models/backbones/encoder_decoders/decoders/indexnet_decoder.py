import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from mmedit.models.common import ConvModule


class IndexedUpsample(nn.Module):
    """Indexed upsample module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int, optional): Kernel size of the convolution layer.
            Defaults to 5.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        conv_module (ConvModule | DepthwiseSeparableConvModule, optional):
            Conv module. Defaults to ConvModule.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 norm_cfg=dict(type='BN'),
                 conv_module=ConvModule):
        super(IndexedUpsample, self).__init__()

        self.conv = conv_module(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6'))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, shortcut, idx_dec=None):
        if idx_dec is not None:
            x = idx_dec * F.interpolate(x, size=shortcut.shape[2:])
        out = torch.cat((x, shortcut), dim=1)
        return self.conv(out)
