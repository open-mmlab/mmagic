'''PyTorch implementation of TOFlow
Paper: Xue et al., Video Enhancement with Task-Oriented Flow, IJCV 2018
Code reference:
1. https://github.com/anchen1011/toflow
2. https://github.com/Coldog2333/pytoflow
'''

import torch
import torch.nn as nn
from .arch_util import flow_warp


def normalize(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).type_as(x)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).type_as(x)
    return (x - mean) / std


def denormalize(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).type_as(x)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).type_as(x)
    return x * std + mean


class SpyNet_Block(nn.Module):
    '''A submodule of SpyNet.'''

    def __init__(self):
        super(SpyNet_Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, x):
        '''
        input: x: [ref im, nbr im, initial flow] - (B, 8, H, W)
        output: estimated flow - (B, 2, H, W)
        '''
        return self.block(x)


class SpyNet(nn.Module):
    '''SpyNet for estimating optical flow
    Ranjan et al., Optical Flow Estimation using a Spatial Pyramid Network, 2016'''

    def __init__(self):
        super(SpyNet, self).__init__()

        self.blocks = nn.ModuleList([SpyNet_Block() for _ in range(4)])

    def forward(self, ref, nbr):
        '''Estimating optical flow in coarse level, upsample, and estimate in fine level
        input: ref: reference image - [B, 3, H, W]
               nbr: the neighboring image to be warped - [B, 3, H, W]
        output: estimated optical flow - [B, 2, H, W]
        '''
        B, C, H, W = ref.size()
        ref = [ref]
        nbr = [nbr]

        for _ in range(3):
            ref.insert(
                0,
                nn.functional.avg_pool2d(input=ref[0], kernel_size=2, stride=2,
                                         count_include_pad=False))
            nbr.insert(
                0,
                nn.functional.avg_pool2d(input=nbr[0], kernel_size=2, stride=2,
                                         count_include_pad=False))

        flow = torch.zeros(B, 2, H // 16, W // 16).type_as(ref[0])

        for i in range(4):
            flow_up = nn.functional.interpolate(input=flow, scale_factor=2, mode='bilinear',
                                                align_corners=True) * 2.0
            flow = flow_up + self.blocks[i](torch.cat(
                [ref[i], flow_warp(nbr[i], flow_up.permute(0, 2, 3, 1)), flow_up], 1))
        return flow


class TOFlow(nn.Module):
    def __init__(self, adapt_official=False):
        super(TOFlow, self).__init__()

        self.SpyNet = SpyNet()

        self.conv_3x7_64_9x9 = nn.Conv2d(3 * 7, 64, 9, 1, 4)
        self.conv_64_64_9x9 = nn.Conv2d(64, 64, 9, 1, 4)
        self.conv_64_64_1x1 = nn.Conv2d(64, 64, 1)
        self.conv_64_3_1x1 = nn.Conv2d(64, 3, 1)

        self.relu = nn.ReLU(inplace=True)

        self.adapt_official = adapt_official  # True if using translated official weights else False

    def forward(self, x):
        """
        input: x: input frames - [B, 7, 3, H, W]
        output: SR reference frame - [B, 3, H, W]
        """

        B, T, C, H, W = x.size()
        x = normalize(x.view(-1, C, H, W)).view(B, T, C, H, W)

        ref_idx = 3
        x_ref = x[:, ref_idx, :, :, :]

        # In the official torch code, the 0-th frame is the reference frame
        if self.adapt_official:
            x = x[:, [3, 0, 1, 2, 4, 5, 6], :, :, :]
            ref_idx = 0

        x_warped = []
        for i in range(7):
            if i == ref_idx:
                x_warped.append(x_ref)
            else:
                x_nbr = x[:, i, :, :, :]
                flow = self.SpyNet(x_ref, x_nbr).permute(0, 2, 3, 1)
                x_warped.append(flow_warp(x_nbr, flow))
        x_warped = torch.stack(x_warped, dim=1)

        x = x_warped.view(B, -1, H, W)
        x = self.relu(self.conv_3x7_64_9x9(x))
        x = self.relu(self.conv_64_64_9x9(x))
        x = self.relu(self.conv_64_64_1x1(x))
        x = self.conv_64_3_1x1(x) + x_ref

        return denormalize(x)
