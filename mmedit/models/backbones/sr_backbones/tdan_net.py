import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init
from mmcv.ops import DeformConv2dPack, deform_conv2d
from mmcv.runner import load_checkpoint
from torch.nn.modules.utils import _pair

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class AugmentedDeformConv2dPack(DeformConv2dPack):
    """Augmented Deformable Convolution Pack.

    Different from DeformConv2dPack, which generates offsets from the
    preceeding feature, this AugmentedDeformConv2dPack takes another feature to
    generate the offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
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
    """

    def __init__(self):

        super().__init__()

        self.feat_extract = nn.Sequential(
            ConvModule(3, 64, 3, padding=1),
            make_layer(ResidualBlockNoBN, 5, mid_channels=64))

        self.feat_aggregate = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=True),
            DeformConv2dPack(64, 64, 3, padding=1, deform_groups=8),
            DeformConv2dPack(64, 64, 3, padding=1, deform_groups=8))
        self.align_1 = AugmentedDeformConv2dPack(
            64, 64, 3, padding=1, deform_groups=8)
        self.align_2 = DeformConv2dPack(64, 64, 3, padding=1, deform_groups=8)
        self.to_rgb = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        self.reconstruct = nn.Sequential(
            ConvModule(3 * 5, 64, 3, padding=1),
            make_layer(ResidualBlockNoBN, 10, mid_channels=64),
            PixelShufflePack(64, 64, 2, upsample_kernel=3),
            PixelShufflePack(64, 64, 2, upsample_kernel=3),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False))

    def forward(self, lrs):
        """Forward function for TDANNet.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR image with shape (n, c, 4h, 4w).
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

        # return HR center frame and the aligned LR frames
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
