# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmedit.registry import MODELS
from .ifrnet_utils import resize, warp


def convrelu(in_channels,
             out_channels,
             kernel_size=3,
             stride=1,
             padding=1,
             dilation=1,
             groups=1,
             bias=True):
    """Conv2d with PReLU activation function."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias), nn.PReLU(out_channels))


class ResBlock(BaseModule):
    """ResBlock with 5 convrelu."""

    def __init__(self, in_channels, side_channels, bias=True, init_cfg=None):
        super().__init__(init_cfg)
        self.side_channels = side_channels
        self.conv1 = convrelu(in_channels, in_channels, bias=bias)
        self.conv2 = convrelu(side_channels, side_channels, bias=bias)
        self.conv3 = convrelu(in_channels, in_channels, bias=bias)
        self.conv4 = convrelu(side_channels, side_channels, bias=bias)
        self.conv5 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out_middle = out[:, :-self.side_channels, :, :]
        out_side = out[:, -self.side_channels:, :, :]
        out_side = self.conv2(out_side)
        out = torch.cat([out_middle, out_side], dim=1)

        out = self.conv3(out)

        out_middle = out[:, :-self.side_channels, :, :]
        out_side = out[:, -self.side_channels:, :, :]
        out_side = self.conv4(out_side)
        out = torch.cat([out_middle, out_side], dim=1)

        out = self.conv5(out)
        out = x + out
        out = self.prelu(out)
        return out


class Encoder(BaseModule):
    """Encoder for IFRNet."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.pyramid1 = nn.Sequential(
            convrelu(3, 32, 3, 2, 1), convrelu(32, 32, 3, 1, 1))
        self.pyramid2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1), convrelu(48, 48, 3, 1, 1))
        self.pyramid3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1), convrelu(72, 72, 3, 1, 1))
        self.pyramid4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1), convrelu(96, 96, 3, 1, 1))

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(BaseModule):
    """fourth Decoder for IFRNet."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(192 + 1, 192), ResBlock(192, 32),
            nn.ConvTranspose2d(192, 76, 4, 2, 1, bias=True))

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(BaseModule):
    """Third Decoder for IFRNet."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.convblock = nn.Sequential(
            convrelu(220, 216), ResBlock(216, 32),
            nn.ConvTranspose2d(216, 52, 4, 2, 1, bias=True))

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(BaseModule):
    """Second Decoder for IFRNet."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.convblock = nn.Sequential(
            convrelu(148, 144), ResBlock(144, 32),
            nn.ConvTranspose2d(144, 36, 4, 2, 1, bias=True))

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(BaseModule):
    """First Decoder for IFRNet."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.convblock = nn.Sequential(
            convrelu(100, 96), ResBlock(96, 32),
            nn.ConvTranspose2d(96, 8, 4, 2, 1, bias=True))

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


@MODELS.register_module()
class IFRNetInterpolator(BaseModule):
    """Base module of IFRNet for video frame interpolation.

    Paper:
        IFRNet: Intermediate Feature Refine Network
                for Efficient Frame Interpolation

    Ref repo: https://github.com/ltkong218/IFRNet
    """

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()

    def forward(self, img0, img1, embt):
        mean_ = torch.cat([img0, img1], 2).mean(
            1, keepdim=True).mean(
                2, keepdim=True).mean(
                    3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)

        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 -
                                              up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0., 1.)

        out = dict(
            pred_img=imgt_pred,
            feats=[ft_1_, ft_2_, ft_3_],
            flows0=[up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
            flows1=[up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4])
        return out
