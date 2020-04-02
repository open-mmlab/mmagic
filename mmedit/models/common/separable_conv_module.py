import torch.nn as nn

from .conv_module import ConvModule


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        kwargs (optional): Arguments for ConvModule. See ConvModule ref.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            **kwargs)

        # pop out config for the depthwise conv and keep norm_cfg etc.
        for key in ['stride', 'padding', 'dilation']:
            kwargs.pop(key, None)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1,
                                         **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
