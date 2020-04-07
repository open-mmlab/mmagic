import torch
import torch.nn as nn
from mmedit.models.common import ConvModule


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

    From https://github.com/poppinace/indexnet_matting.

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
                 kernel_size=2,
                 padding=0,
                 norm_cfg=dict(type='BN'),
                 use_nonlinear=False):
        super(HolisticIndexBlock, self).__init__()

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

    From https://github.com/poppinace/indexnet_matting.

    Args:
        in_channels (int): Input channels of the holistic index block.
        kernel_size (int): Kernel size of the conv layers. Default: 2.
        padding (int): Padding number of the conv layers. Default: 0.
        groups (int): Groups of conv layers. When `groups` equals 1, it
            corresponds to the `m2o` mode in the paper. When `groups` equals
            `in_channels`, it corresponds to the `o2o` mode in the paper.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        use_nonlinear (bool): Whether add a non-linear conv layer in the index
            blocks. Default: False.
    """

    def __init__(self,
                 in_channels,
                 kernel_size=2,
                 padding=0,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 use_nonlinear=False):
        super(DepthwiseIndexBlock, self).__init__()

        index_blocks = []
        for i in range(4):
            index_blocks.append(
                build_index_block(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=2,
                    padding=padding,
                    groups=groups,
                    norm_cfg=norm_cfg,
                    use_nonlinear=use_nonlinear))
        self.index_blocks = nn.ModuleList(index_blocks)

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
