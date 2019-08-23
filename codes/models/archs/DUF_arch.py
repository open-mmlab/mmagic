'''Network architecture for DUF:
Deep Video Super-Resolution Network Using Dynamic Upsampling Filters
Without Explicit Motion Compensation (CVPR18)
https://github.com/yhjo09/VSR-DUF

For all the models below, [adapt_official] is only necessary when
loading the weights converted from the official TensorFlow weights.
Please set it to [False] if you are training the model from scratch.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def adapt_official(Rx, scale=4):
    '''Adapt the weights translated from the official tensorflow weights
    Not necessary if you are training from scratch'''
    x = Rx.clone()
    x1 = x[:, ::3, :, :]
    x2 = x[:, 1::3, :, :]
    x3 = x[:, 2::3, :, :]

    Rx[:, :scale**2, :, :] = x1
    Rx[:, scale**2:2 * (scale**2), :, :] = x2
    Rx[:, 2 * (scale**2):, :, :] = x3

    return Rx


class DenseBlock(nn.Module):
    '''Dense block
    for the second denseblock, t_reduced = True'''

    def __init__(self, nf=64, ng=32, t_reduce=False):
        super(DenseBlock, self).__init__()
        self.t_reduce = t_reduce
        if self.t_reduce:
            pad = (0, 1, 1)
        else:
            pad = (1, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)
        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng, nf + ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                  bias=True)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)
        self.bn3d_5 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_5 = nn.Conv3d(nf + 2 * ng, nf + 2 * ng, (1, 1, 1), stride=(1, 1, 1),
                                  padding=(0, 0, 0), bias=True)
        self.bn3d_6 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_6 = nn.Conv3d(nf + 2 * ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad,
                                  bias=True)

    def forward(self, x):
        '''x: [B, C, T, H, W]
        C: nf -> nf + 3 * ng
        T: 1) 7 -> 7 (t_reduce=False);
           2) 7 -> 7 - 2 * 3 = 1 (t_reduce=True)'''
        x1 = self.conv3d_1(F.relu(self.bn3d_1(x), inplace=True))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1), inplace=True))
        if self.t_reduce:
            x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)
        else:
            x1 = torch.cat((x, x1), 1)

        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1), inplace=True))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2), inplace=True))
        if self.t_reduce:
            x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)
        else:
            x2 = torch.cat((x1, x2), 1)

        x3 = self.conv3d_5(F.relu(self.bn3d_5(x2), inplace=True))
        x3 = self.conv3d_6(F.relu(self.bn3d_6(x3), inplace=True))
        if self.t_reduce:
            x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)
        else:
            x3 = torch.cat((x2, x3), 1)
        return x3


class DynamicUpsamplingFilter_3C(nn.Module):
    '''dynamic upsampling filter with 3 channels applying the same filters
    filter_size: filter size of the generated filters, shape (C, kH, kW)'''

    def __init__(self, filter_size=(1, 5, 5)):
        super(DynamicUpsamplingFilter_3C, self).__init__()
        # generate a local expansion filter, used similar to im2col
        nF = np.prod(filter_size)
        expand_filter_np = np.reshape(np.eye(nF, nF),
                                      (nF, filter_size[0], filter_size[1], filter_size[2]))
        expand_filter = torch.from_numpy(expand_filter_np).float()
        self.expand_filter = torch.cat((expand_filter, expand_filter, expand_filter),
                                       0)  # [75, 1, 5, 5]

    def forward(self, x, filters):
        '''x: input image, [B, 3, H, W]
        filters: generate dynamic filters, [B, F, R, H, W], e.g., [B, 25, 16, H, W]
            F: prod of filter kernel size, e.g., 5*5 = 25
            R: used for upsampling, similar to pixel shuffle, e.g., 4*4 = 16 for x4
        Return: filtered image, [B, 3*R, H, W]
        '''
        B, nF, R, H, W = filters.size()
        # using group convolution
        input_expand = F.conv2d(x, self.expand_filter.type_as(x), padding=2,
                                groups=3)  # [B, 75, H, W] similar to im2col
        input_expand = input_expand.view(B, 3, nF, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 25]
        filters = filters.permute(0, 3, 4, 1, 2)  # [B, H, W, 25, 16]
        out = torch.matmul(input_expand, filters)  # [B, H, W, 3, 16]
        return out.permute(0, 3, 4, 1, 2).view(B, 3 * R, H, W)  # [B, 3*16, H, W]


class DUF_16L(nn.Module):
    '''Official DUF structure with 16 layers'''

    def __init__(self, scale=4, adapt_official=False):
        super(DUF_16L, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dense_block_1 = DenseBlock(64, 64 // 2, t_reduce=False)  # 64 + 32 * 3 = 160, T = 7
        self.dense_block_2 = DenseBlock(160, 64 // 2, t_reduce=True)  # 160 + 32 * 3 = 256, T = 1
        self.bn3d_2 = nn.BatchNorm3d(256, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(256, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                                  bias=True)

        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_r2 = nn.Conv3d(256, 3 * (scale**2), (1, 1, 1), stride=(1, 1, 1),
                                   padding=(0, 0, 0), bias=True)

        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_f2 = nn.Conv3d(512, 1 * 5 * 5 * (scale**2), (1, 1, 1), stride=(1, 1, 1),
                                   padding=(0, 0, 0), bias=True)

        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))

        self.scale = scale
        self.adapt_official = adapt_official

    def forward(self, x):
        '''
        x: [B, T, C, H, W], T = 7. reshape to [B, C, T, H, W] for Conv3D
        Generate filters and image residual:
        Fx: [B, 25, 16, H, W] for DynamicUpsamplingFilter_3C
        Rx: [B, 3*16, 1, H, W]
        '''
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for Conv3D
        x_center = x[:, :, T // 2, :, :]

        x = self.conv3d_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)  # reduce T to 1
        x = F.relu(self.conv3d_2(F.relu(self.bn3d_2(x), inplace=True)), inplace=True)

        # image residual
        Rx = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))  # [B, 3*16, 1, H, W]

        # filter
        Fx = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))  # [B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 25, self.scale**2, H, W), dim=1)

        # Adapt to official model weights
        if self.adapt_official:
            adapt_official(Rx, scale=self.scale)

        # dynamic filter
        out = self.dynamic_filter(x_center, Fx)  # [B, 3*R, H, W]
        out += Rx.squeeze_(2)
        out = F.pixel_shuffle(out, self.scale)  # [B, 3, H, W]

        return out


class DenseBlock_28L(nn.Module):
    '''The first part of the dense blocks used in DUF_28L
    Temporal dimension remains the same here'''

    def __init__(self, nf=64, ng=16):
        super(DenseBlock_28L, self).__init__()
        pad = (1, 1, 1)

        dense_block_l = []
        for i in range(0, 9):
            dense_block_l.append(nn.BatchNorm3d(nf + i * ng, eps=1e-3, momentum=1e-3))
            dense_block_l.append(nn.ReLU())
            dense_block_l.append(
                nn.Conv3d(nf + i * ng, nf + i * ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                          bias=True))

            dense_block_l.append(nn.BatchNorm3d(nf + i * ng, eps=1e-3, momentum=1e-3))
            dense_block_l.append(nn.ReLU())
            dense_block_l.append(
                nn.Conv3d(nf + i * ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True))

        self.dense_blocks = nn.ModuleList(dense_block_l)

    def forward(self, x):
        '''x: [B, C, T, H, W]
        C: 1) 64 -> 208;
        T: 1) 7 -> 7; (t_reduce=True)'''
        for i in range(0, len(self.dense_blocks), 6):
            y = x
            for j in range(6):
                y = self.dense_blocks[i + j](y)
            x = torch.cat((x, y), 1)
        return x


class DUF_28L(nn.Module):
    '''Official DUF structure with 28 layers'''

    def __init__(self, scale=4, adapt_official=False):
        super(DUF_28L, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dense_block_1 = DenseBlock_28L(64, 16)  # 64 + 16 * 9 = 208, T = 7
        self.dense_block_2 = DenseBlock(208, 16, t_reduce=True)  # 208 + 16 * 3 = 256, T = 1
        self.bn3d_2 = nn.BatchNorm3d(256, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(256, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                                  bias=True)

        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_r2 = nn.Conv3d(256, 3 * (scale**2), (1, 1, 1), stride=(1, 1, 1),
                                   padding=(0, 0, 0), bias=True)

        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_f2 = nn.Conv3d(512, 1 * 5 * 5 * (scale**2), (1, 1, 1), stride=(1, 1, 1),
                                   padding=(0, 0, 0), bias=True)

        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))

        self.scale = scale
        self.adapt_official = adapt_official

    def forward(self, x):
        '''
        x: [B, T, C, H, W], T = 7. reshape to [B, C, T, H, W] for Conv3D
        Generate filters and image residual:
        Fx: [B, 25, 16, H, W] for DynamicUpsamplingFilter_3C
        Rx: [B, 3*16, 1, H, W]
        '''
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] for Conv3D
        x_center = x[:, :, T // 2, :, :]
        x = self.conv3d_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)  # reduce T to 1
        x = F.relu(self.conv3d_2(F.relu(self.bn3d_2(x), inplace=True)), inplace=True)

        # image residual
        Rx = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))  # [B, 3*16, 1, H, W]

        # filter
        Fx = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))  # [B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 25, self.scale**2, H, W), dim=1)

        # Adapt to official model weights
        if self.adapt_official:
            adapt_official(Rx, scale=self.scale)

        # dynamic filter
        out = self.dynamic_filter(x_center, Fx)  # [B, 3*R, H, W]
        out += Rx.squeeze_(2)
        out = F.pixel_shuffle(out, self.scale)  # [B, 3, H, W]
        return out


class DenseBlock_52L(nn.Module):
    '''The first part of the dense blocks used in DUF_52L
    Temporal dimension remains the same here'''

    def __init__(self, nf=64, ng=16):
        super(DenseBlock_52L, self).__init__()
        pad = (1, 1, 1)

        dense_block_l = []
        for i in range(0, 21):
            dense_block_l.append(nn.BatchNorm3d(nf + i * ng, eps=1e-3, momentum=1e-3))
            dense_block_l.append(nn.ReLU())
            dense_block_l.append(
                nn.Conv3d(nf + i * ng, nf + i * ng, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                          bias=True))

            dense_block_l.append(nn.BatchNorm3d(nf + i * ng, eps=1e-3, momentum=1e-3))
            dense_block_l.append(nn.ReLU())
            dense_block_l.append(
                nn.Conv3d(nf + i * ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True))

        self.dense_blocks = nn.ModuleList(dense_block_l)

    def forward(self, x):
        '''x: [B, C, T, H, W]
        C: 1) 64 -> 400;
        T: 1) 7 -> 7; (t_reduce=True)'''
        for i in range(0, len(self.dense_blocks), 6):
            y = x
            for j in range(6):
                y = self.dense_blocks[i + j](y)
            x = torch.cat((x, y), 1)
        return x


class DUF_52L(nn.Module):
    '''Official DUF structure with 52 layers'''

    def __init__(self, scale=4, adapt_official=False):
        super(DUF_52L, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dense_block_1 = DenseBlock_52L(64, 16)  # 64 + 21 * 9 = 400, T = 7
        self.dense_block_2 = DenseBlock(400, 16, t_reduce=True)  # 400 + 16 * 3 = 448, T = 1

        self.bn3d_2 = nn.BatchNorm3d(448, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(448, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                                  bias=True)

        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_r2 = nn.Conv3d(256, 3 * (scale**2), (1, 1, 1), stride=(1, 1, 1),
                                   padding=(0, 0, 0), bias=True)

        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_f2 = nn.Conv3d(512, 1 * 5 * 5 * (scale**2), (1, 1, 1), stride=(1, 1, 1),
                                   padding=(0, 0, 0), bias=True)

        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))

        self.scale = scale
        self.adapt_official = adapt_official

    def forward(self, x):
        '''
        x: [B, T, C, H, W], T = 7. reshape to [B, C, T, H, W] for Conv3D
        Generate filters and image residual:
        Fx: [B, 25, 16, H, W] for DynamicUpsamplingFilter_3C
        Rx: [B, 3*16, 1, H, W]
        '''
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] for Conv3D
        x_center = x[:, :, T // 2, :, :]
        x = self.conv3d_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = F.relu(self.conv3d_2(F.relu(self.bn3d_2(x), inplace=True)), inplace=True)

        # image residual
        Rx = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))  # [B, 3*16, 1, H, W]

        # filter
        Fx = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))  # [B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 25, self.scale**2, H, W), dim=1)

        # Adapt to official model weights
        if self.adapt_official:
            adapt_official(Rx, scale=self.scale)

        # dynamic filter
        out = self.dynamic_filter(x_center, Fx)  # [B, 3*R, H, W]
        out += Rx.squeeze_(2)
        out = F.pixel_shuffle(out, self.scale)  # [B, 3, H, W]
        return out
