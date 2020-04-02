from mmedit.models.common import ConvModule

from ..encoders.resnet import BasicBlock


class BasicBlockDec(BasicBlock):
    """Basic residual block for decoder.

    For decoder, we use ConvTranspose2d with kernel_size 4 and padding 1 for
    conv1. And the output channel of conv1 is modified from `out_channels` to
    `in_channels`.
    """

    def build_conv1(self, in_channels, out_channels, kernel_size, stride,
                    conv_cfg, norm_cfg, act_cfg):
        return ConvModule(
            in_channels,
            in_channels,
            4,
            stride=stride,
            padding=1,
            conv_cfg=dict(type='Deconv'),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def build_conv2(self, in_channels, out_channels, kernel_size, conv_cfg,
                    norm_cfg):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
