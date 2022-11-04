# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.ops import modulated_deform_conv2d
from torch.nn.modules.utils import _pair


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


class DCN_layer(nn.Module):
    """Deformable Convolution (DCN) layer.

    Args:
        in_channels: Dimension of input channels
        out_channels: Dimension of output channels
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True,
                 extra_offset_mask=True):
        super(DCN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels * 2,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init_offset()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter):
        feat_degradation = torch.cat([input_feat, inter], dim=1)

        out = self.conv_offset_mask(feat_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(input_feat.contiguous(), offset, mask,
                                       self.weight, self.bias, self.stride,
                                       self.padding, self.dilation,
                                       self.groups, self.deformable_groups)


class SFT_layer(nn.Module):
    """Spatial Feature Transform (SFT) layer.

    Args:
        in_channels: Dimension of input channels
        out_channels: Dimension of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map:
                                              B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta


class DGM(nn.Module):
    """Degradation-Guided Modules (DGM).

    Under the guidance of degradation z.

    Args:
        in_channels: Dimension of input channels
        out_channels: Dimension of output channels
        kernel_size: Kernel size for DCN
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DGM, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(
            self.in_channels,
            self.out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False)
        self.sft = SFT_layer(self.in_channels, self.out_channels)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter)
        out = dcn_out + sft_out
        out = x + out

        return out


class DGB(nn.Module):
    """Degradation-Guided Blocks (DGB).

    Each DGB contains 2 DGM.

    Args:
        conv (nn.Module): Convolution module.
        n_feat (int): Number of features.
        kernel_size (int): Kernel size for DCN.
    """

    def __init__(self, conv, n_feat, kernel_size):
        super(DGB, self).__init__()

        # self.da_conv1 = DGM(n_feat, n_feat, kernel_size)
        # self.da_conv2 = DGM(n_feat, n_feat, kernel_size)
        self.dgm1 = DGM(n_feat, n_feat, kernel_size)
        self.dgm2 = DGM(n_feat, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''

        out = self.relu(self.dgm1(x, inter))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dgm2(out, inter))
        out = self.conv2(out) + x

        return out


class DGG(nn.Module):
    """DegradationGuided Groups (DGG)

    Basic unit block for DGRN.

    Args:
        conv (nn.Module): Convolution module.
        n_feat (int): Number of features.
        kernel_size (int): Kernel size for DCN.
        n_blocks (int): Number of DGB blocks.
    """

    def __init__(self, conv, n_feat, kernel_size, n_blocks):
        super(DGG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DGB(conv, n_feat, kernel_size) for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        return res
