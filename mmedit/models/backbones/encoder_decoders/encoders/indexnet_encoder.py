import torch
import torch.nn as nn
import torch.nn.functional as F
from mmedit.models.common import ConvModule, DepthwiseSeparableConvModule


def build_index_block(in_channels,
                      out_channels,
                      kernel_size,
                      stride=2,
                      padding=0,
                      groups=1,
                      norm_cfg=dict(type='BN'),
                      use_nonlinear=False,
                      expansion=1):
    if use_nonlinear:
        return nn.Sequential(
            ConvModule(
                in_channels,
                in_channels * expansion,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            ConvModule(
                in_channels * expansion,
                out_channels,
                1,
                stride=1,
                padding=0,
                groups=groups,
                bias=False,
                norm_cfg=None,
                act_cfg=None))
    else:
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            norm_cfg=None,
            act_cfg=None)


class HolisticIndexBlock(nn.Module):
    """Holistic Index Block.

    From https://arxiv.org/abs/1908.00672.

    Args:
        in_channels (int): Input channels of the holistic index block.
        kernel_size (int): Kernel size of the conv layers. Default: 2.
        padding (int): Padding number of the conv layers. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        use_nonlinear (bool): Whether add a non-linear conv layer in the index
            block. Default: False.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='BN'),
                 use_context=False,
                 use_nonlinear=False):
        super(HolisticIndexBlock, self).__init__()

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        self.index_block = build_index_block(
            in_channels,
            4,
            kernel_size,
            stride=2,
            padding=padding,
            groups=1,
            norm_cfg=norm_cfg,
            use_nonlinear=use_nonlinear,
            expansion=2)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.index_block(x)

        # normalization
        y = self.sigmoid(x)
        z = self.softmax(y)
        # pixel shuffling
        idx_enc = self.pixel_shuffle(z)
        idx_dec = self.pixel_shuffle(y)

        return idx_enc, idx_dec


class DepthwiseIndexBlock(nn.Module):
    """Depthwise index block.

    From https://arxiv.org/abs/1908.00672.

    Args:
        in_channels (int): Input channels of the holistic index block.
        kernel_size (int): Kernel size of the conv layers. Default: 2.
        padding (int): Padding number of the conv layers. Default: 0.
        mode (str): Mode of index block. Should be 'o2o' or 'm2o'. In 'o2o'
            mode, the group of the conv layers is 1; In 'm2o' mode, the group
            of the conv layer is `in_channels`.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        use_nonlinear (bool): Whether add a non-linear conv layer in the index
            blocks. Default: False.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='BN'),
                 use_context=False,
                 use_nonlinear=False,
                 mode='o2o'):
        super(DepthwiseIndexBlock, self).__init__()

        groups = in_channels if mode == 'o2o' else 1

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        self.index_blocks = nn.ModuleList()
        for i in range(4):
            self.index_blocks.append(
                build_index_block(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=2,
                    padding=padding,
                    groups=groups,
                    norm_cfg=norm_cfg,
                    use_nonlinear=use_nonlinear))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        n, c, h, w = x.shape

        feature_list = [
            _index_block(x).unsqueeze(2) for _index_block in self.index_blocks
        ]
        x = torch.cat(feature_list, dim=2)

        # normalization
        y = self.sigmoid(x)
        z = self.softmax(y)
        # pixel shuffling
        y = y.view(n, c * 4, h // 2, w // 2)
        z = z.view(n, c * 4, h // 2, w // 2)
        idx_enc = self.pixel_shuffle(z)
        idx_dec = self.pixel_shuffle(y)

        return idx_enc, idx_dec


class InvertedResidual(nn.Module):
    """Inverted residual layer for indexnet encoder.

    It basicaly is a depthwise separable conv module. If `expand_ratio` is not
    one, then a conv module of kernel_size 1 will be inserted to change the
    input channels to `in_channels * expand_ratio`.

    Args:
        in_channels (int): Input channels of the layer.
        out_channels (int): Output channels of the layer.
        stride (int): Stride of the depthwise separable conv module.
        dilation (int): Dilation of the depthwise separable conv module.
        expand_ratio (float): Expand ratio of the input channels of the
            depthwise separable conv module.
        norm_cfg (dict | None): Config dict for normalization layer.
        use_res_connect (bool, optional): Whether use shortcut connection.
            Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation,
                 expand_ratio,
                 norm_cfg,
                 use_res_connect=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], 'stride must 1 or 2'

        self.use_res_connect = use_res_connect
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = DepthwiseSeparableConvModule(
                in_channels,
                out_channels,
                3,
                stride=stride,
                dilation=dilation,
                norm_cfg=norm_cfg,
                dw_act_cfg=dict(type='ReLU6'),
                pw_act_cfg=None)
        else:
            hidden_dim = round(in_channels * expand_ratio)
            self.conv = nn.Sequential(
                ConvModule(
                    in_channels,
                    hidden_dim,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                DepthwiseSeparableConvModule(
                    hidden_dim,
                    out_channels,
                    3,
                    stride=stride,
                    dilation=dilation,
                    norm_cfg=norm_cfg,
                    dw_act_cfg=dict(type='ReLU6'),
                    pw_act_cfg=None))

    def pad(self, inputs, kernel_size, dilation):
        effective_ksize = kernel_size + (kernel_size - 1) * (dilation - 1)
        left = (effective_ksize - 1) // 2
        right = effective_ksize // 2
        return F.pad(inputs, (left, right, left, right))

    def forward(self, x):
        out = self.conv(self.pad(x, self.kernel_size, self.dilation))

        if self.use_res_connect:
            out = out + x

        return out
