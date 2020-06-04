import torch.nn as nn
from mmcv.cnn import ConvModule
from mmedit.models.registry import COMPONENTS


class ResidualDilationBlock(nn.Module):
    """Residual dilation block.

    In this module, we modify the basic block in ResNet with dilated conv.

    Args:
        in_channels (int): Channels of input feature or image.
        out_channels (int): Channels of output feature.
        kernel_size (int or tuple[int]): Same as nn.Conv2d. Defaults to 3.
        stride (int or tuple[int]): Same as nn.Conv2d. Defaults to 1.
        dilation (int, optional): Same as nn.Conv2d. Defaults to 2.
        norm_cfg (None | dict, optional): Config dict for normalization layer.
            Defaults to None.
        act_cfg (None | dict, optional): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.2).
        kwargs (keyword arguments): Keyword arguments for `ConvModule`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=2,
                 norm_cfg=None,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 **kwargs):
        super(ResidualDilationBlock, self).__init__()
        self.with_skip_conv = in_channels != out_channels

        padding_ = (kernel_size - 1) // 2 * dilation
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_,
            stride=stride,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        padding_ = (kernel_size - 1) // 2
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_,
            stride=stride,
            dilation=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        if self.with_skip_conv:
            self.skip_conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                act_cfg=None,
                norm_cfg=norm_cfg,
                **kwargs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.with_skip_conv:
            x = self.skip_conv(x)

        x = x + out

        return x


@COMPONENTS.register_module
class TMADDilationNeck(nn.Module):
    """TMAD dilation neck.

    Args:
        in_channels (int): Channels of input feature or image.
        out_channels (int): Channels of output feature.
        num_blocks (int, optional): The number of dilation blocks.
            Defaults to 3.
        dilation (int, optional): Same as nn.Conv2d. Defaults to 2.
        kwargs (keyword arguments): Keyword arguments for `ConvModule`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 dilation=2,
                 **kwargs):
        super(TMADDilationNeck, self).__init__()
        self.num_blocks = num_blocks

        dilation_blocks_ = []
        for i in range(num_blocks):
            dilation_blocks_.append(
                ResidualDilationBlock(
                    in_channels, out_channels, dilation=dilation, **kwargs))
            in_channels = out_channels

        self.dilation_blocks = nn.ModuleList(dilation_blocks_)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['out']

        output_dict = dict()
        for i in range(self.num_blocks):
            x = self.dilation_blocks[i](x)
            output_dict[f'dilation{i}'] = x
        output_dict['out'] = x

        return output_dict
