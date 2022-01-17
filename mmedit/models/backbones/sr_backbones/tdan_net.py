# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init
from mmcv.ops import DeformConv2d, DeformConv2dPack, deform_conv2d
from mmcv.runner import load_checkpoint
from torch.nn.modules.utils import _pair

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class AugmentedDeformConv2dPack(DeformConv2d):
    """Augmented Deformable Convolution Pack.

    Different from DeformConv2dPack, which generates offsets from the
    preceding feature, this AugmentedDeformConv2dPack takes another feature to
    generate the offsets.

    Args:
        in_channels (int): Number of channels in the input feature.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple[int]): Size of the convolving kernel.
        stride (int or tuple[int]): Stride of the convolution. Default: 1.
        padding (int or tuple[int]): Zero-padding added to both sides of the
            input. Default: 0.
        dilation (int or tuple[int]): Spacing between kernel elements.
            Default: 1.
        groups (int): Number of blocked connections from input channels to
            output channels. Default: 1.
        deform_groups (int): Number of deformable group partitions.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        offset = self.conv_offset(extra_feat)
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)


@BACKBONES.register_module()
class TDANNet(nn.Module):
    """TDAN network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        TDAN: Temporally-Deformable Alignment Network for Video Super-
        Resolution, CVPR, 2020

    Args:
        in_channels (int): Number of channels of the input image. Default: 3.
        mid_channels (int): Number of channels of the intermediate features.
            Default: 64.
        out_channels (int): Number of channels of the output image. Default: 3.
        num_blocks_before_align (int): Number of residual blocks before
            temporal alignment. Default: 5.
        num_blocks_before_align (int): Number of residual blocks after
            temporal alignment. Default: 10.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 out_channels=3,
                 num_blocks_before_align=5,
                 num_blocks_after_align=10):

        super().__init__()

        self.feat_extract = nn.Sequential(
            ConvModule(in_channels, mid_channels, 3, padding=1),
            make_layer(
                ResidualBlockNoBN,
                num_blocks_before_align,
                mid_channels=mid_channels))

        self.feat_aggregate = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1, bias=True),
            DeformConv2dPack(
                mid_channels, mid_channels, 3, padding=1, deform_groups=8),
            DeformConv2dPack(
                mid_channels, mid_channels, 3, padding=1, deform_groups=8))
        self.align_1 = AugmentedDeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=8)
        self.align_2 = DeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=8)
        self.to_rgb = nn.Conv2d(mid_channels, 3, 3, padding=1, bias=True)

        self.reconstruct = nn.Sequential(
            ConvModule(in_channels * 5, mid_channels, 3, padding=1),
            make_layer(
                ResidualBlockNoBN,
                num_blocks_after_align,
                mid_channels=mid_channels),
            PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
            PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False))

    def forward(self, lrs):
        """Forward function for TDANNet.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            tuple[Tensor]: Output HR image with shape (n, c, 4h, 4w) and
            aligned LR images with shape (n, t, c, h, w).
        """
        n, t, c, h, w = lrs.size()
        lr_center = lrs[:, t // 2, :, :, :]  # LR center frame

        # extract features
        feats = self.feat_extract(lrs.view(-1, c, h, w)).view(n, t, -1, h, w)

        # alignment of LR frames
        feat_center = feats[:, t // 2, :, :, :].contiguous()
        aligned_lrs = []
        for i in range(0, t):
            if i == t // 2:
                aligned_lrs.append(lr_center)
            else:
                feat_neig = feats[:, i, :, :, :].contiguous()
                feat_agg = torch.cat([feat_center, feat_neig], dim=1)
                feat_agg = self.feat_aggregate(feat_agg)

                aligned_feat = self.align_2(self.align_1(feat_neig, feat_agg))
                aligned_lrs.append(self.to_rgb(aligned_feat))

        aligned_lrs = torch.cat(aligned_lrs, dim=1)

        # output HR center frame and the aligned LR frames
        return self.reconstruct(aligned_lrs), aligned_lrs.view(n, t, c, h, w)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
