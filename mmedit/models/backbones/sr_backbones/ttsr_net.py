from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

# Use partial to specify some default arguments
_conv3x3_layer = partial(
    build_conv_layer, dict(type='Conv2d'), kernel_size=3, padding=1)
_conv1x1_layer = partial(
    build_conv_layer, dict(type='Conv2d'), kernel_size=1, padding=0)


class SFE(nn.Module):
    """Structural Feature Encoder.

    Backbone of Texture Transformer Network for Image Super-Resolution.

    Args:
        in_channels (int): Number of channels in the input image
        mid_channels (int): Channel number of intermediate features
        num_blocks (int): Block number in the trunk network
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, res_scale):
        super().__init__()

        self.num_blocks = num_blocks
        self.conv_first = _conv3x3_layer(in_channels, mid_channels)

        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last = _conv3x3_layer(mid_channels, mid_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x1 = x = F.relu(self.conv_first(x))
        x = self.body(x)
        x = self.conv_last(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    """Cross-Scale Feature Integration between 1x and 2x features.

    Cross-Scale Feature Integration in Texture Transformer Network for
        Image Super-Resolution.
    It is cross-scale feature integration between 1x and 2x features.
        For example, `conv2to1` means conv layer from 2x feature to 1x
        feature. Down-sampling is achieved by conv layer with stride=2,
        and up-sampling is achieved by bicubic interpolate and conv layer.

    Args:
        mid_channels (int): Channel number of intermediate features
    """

    def __init__(self, mid_channels):
        super().__init__()
        self.conv1to2 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv2to1 = _conv3x3_layer(mid_channels, mid_channels, stride=2)

        self.conv_merge1 = _conv3x3_layer(mid_channels * 2, mid_channels)
        self.conv_merge2 = _conv3x3_layer(mid_channels * 2, mid_channels)

    def forward(self, x1, x2):
        """Forward function.

        Args:
            x1 (Tensor): Input tensor with shape (n, c, h, w).
            x2 (Tensor): Input tensor with shape (n, c, 2h, 2w).

        Returns:
            x1 (Tensor): Output tensor with shape (n, c, h, w).
            x2 (Tensor): Output tensor with shape (n, c, 2h, 2w).
        """

        x12 = F.interpolate(
            x1, scale_factor=2, mode='bicubic', align_corners=False)
        x12 = F.relu(self.conv1to2(x12))
        x21 = F.relu(self.conv2to1(x2))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12), dim=1)))

        return x1, x2


class CSFI3(nn.Module):
    """Cross-Scale Feature Integration between 1x, 2x, and 4x features.

    Cross-Scale Feature Integration in Texture Transformer Network for
        Image Super-Resolution.
    It is cross-scale feature integration between 1x and 2x features.
        For example, `conv2to1` means conv layer from 2x feature to 1x
        feature. Down-sampling is achieved by conv layer with stride=2,
        and up-sampling is achieved by bicubic interpolate and conv layer.

    Args:
        mid_channels (int): Channel number of intermediate features
    """

    def __init__(self, mid_channels):
        super().__init__()
        self.conv1to2 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv1to4 = _conv1x1_layer(mid_channels, mid_channels)

        self.conv2to1 = _conv3x3_layer(mid_channels, mid_channels, stride=2)
        self.conv2to4 = _conv1x1_layer(mid_channels, mid_channels)

        self.conv4to1_1 = _conv3x3_layer(mid_channels, mid_channels, stride=2)
        self.conv4to1_2 = _conv3x3_layer(mid_channels, mid_channels, stride=2)
        self.conv4to2 = _conv3x3_layer(mid_channels, mid_channels, stride=2)

        self.conv_merge1 = _conv3x3_layer(mid_channels * 3, mid_channels)
        self.conv_merge2 = _conv3x3_layer(mid_channels * 3, mid_channels)
        self.conv_merge4 = _conv3x3_layer(mid_channels * 3, mid_channels)

    def forward(self, x1, x2, x4):
        """Forward function.

        Args:
            x1 (Tensor): Input tensor with shape (n, c, h, w).
            x2 (Tensor): Input tensor with shape (n, c, 2h, 2w).
            x4 (Tensor): Input tensor with shape (n, c, 4h, 4w).

        Returns:
            x1 (Tensor): Output tensor with shape (n, c, h, w).
            x2 (Tensor): Output tensor with shape (n, c, 2h, 2w).
            x4 (Tensor): Output tensor with shape (n, c, 4h, 4w).
        """

        x12 = F.interpolate(
            x1, scale_factor=2, mode='bicubic', align_corners=False)
        x12 = F.relu(self.conv1to2(x12))
        x14 = F.interpolate(
            x1, scale_factor=4, mode='bicubic', align_corners=False)
        x14 = F.relu(self.conv1to4(x14))

        x21 = F.relu(self.conv2to1(x2))
        x24 = F.interpolate(
            x2, scale_factor=2, mode='bicubic', align_corners=False)
        x24 = F.relu(self.conv2to4(x24))

        x41 = F.relu(self.conv4to1_1(x4))
        x41 = F.relu(self.conv4to1_2(x41))
        x42 = F.relu(self.conv4to2(x4))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x41), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x42), dim=1)))
        x4 = F.relu(self.conv_merge4(torch.cat((x4, x14, x24), dim=1)))

        return x1, x2, x4


class MergeFeatures(nn.Module):
    """Merge Features. Merge 1x, 2x, and 4x features.

    Final module of Texture Transformer Network for Image Super-Resolution.

    Args:
        mid_channels (int): Channel number of intermediate features
        out_channels (int): Number of channels in the output image
    """

    def __init__(self, mid_channels, out_channels):
        super().__init__()
        self.conv1to4 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv2to4 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv_merge = _conv3x3_layer(mid_channels * 3, mid_channels)
        self.conv_last1 = _conv3x3_layer(mid_channels, mid_channels // 2)
        self.conv_last2 = _conv1x1_layer(mid_channels // 2, out_channels)

    def forward(self, x1, x2, x4):
        """Forward function.

        Args:
            x1 (Tensor): Input tensor with shape (n, c, h, w).
            x2 (Tensor): Input tensor with shape (n, c, 2h, 2w).
            x4 (Tensor): Input tensor with shape (n, c, 4h, 4w).

        Returns:
            x (Tensor): Output tensor with shape (n, c_out, 4h, 4w).
        """

        x14 = F.interpolate(
            x1, scale_factor=4, mode='bicubic', align_corners=False)
        x14 = F.relu(self.conv1to4(x14))
        x24 = F.interpolate(
            x2, scale_factor=2, mode='bicubic', align_corners=False)
        x24 = F.relu(self.conv2to4(x24))

        x = F.relu(self.conv_merge(torch.cat((x4, x14, x24), dim=1)))
        x = self.conv_last1(x)
        x = self.conv_last2(x)
        x = torch.clamp(x, -1, 1)

        return x


@BACKBONES.register_module()
class TTSRNet(nn.Module):
    """TTSR network structure (main-net) for reference-based super-resolution.

    Paper: Learning Texture Transformer Network for Image Super-Resolution

    Adapted from 'https://github.com/researchmm/TTSR.git'
    'https://github.com/researchmm/TTSR'
    Copyright permission at 'https://github.com/researchmm/TTSR/issues/38'.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels in the output image
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (tuple[int]): Block numbers in the trunk network.
            Default: (16, 16, 8, 4)
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 texture_channels=64,
                 num_blocks=(16, 16, 8, 4),
                 res_scale=1.0):
        super().__init__()

        self.texture_channels = texture_channels

        self.sfe = SFE(in_channels, mid_channels, num_blocks[0], res_scale)

        # stage 1
        self.conv_first1 = _conv3x3_layer(4 * texture_channels + mid_channels,
                                          mid_channels)

        self.res_block1 = make_layer(
            ResidualBlockNoBN,
            num_blocks[1],
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last1 = _conv3x3_layer(mid_channels, mid_channels)

        # up-sampling 1 -> 2
        self.up1 = PixelShufflePack(
            in_channels=mid_channels,
            out_channels=mid_channels,
            scale_factor=2,
            upsample_kernel=3)

        # stage 2
        self.conv_first2 = _conv3x3_layer(2 * texture_channels + mid_channels,
                                          mid_channels)

        self.csfi2 = CSFI2(mid_channels)

        self.res_block2_1 = make_layer(
            ResidualBlockNoBN,
            num_blocks[2],
            mid_channels=mid_channels,
            res_scale=res_scale)
        self.res_block2_2 = make_layer(
            ResidualBlockNoBN,
            num_blocks[2],
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last2_1 = _conv3x3_layer(mid_channels, mid_channels)
        self.conv_last2_2 = _conv3x3_layer(mid_channels, mid_channels)

        # up-sampling 2 -> 3
        self.up2 = PixelShufflePack(
            in_channels=mid_channels,
            out_channels=mid_channels,
            scale_factor=2,
            upsample_kernel=3)

        # stage 3
        self.conv_first3 = _conv3x3_layer(texture_channels + mid_channels,
                                          mid_channels)

        self.csfi3 = CSFI3(mid_channels)

        self.res_block3_1 = make_layer(
            ResidualBlockNoBN,
            num_blocks[3],
            mid_channels=mid_channels,
            res_scale=res_scale)
        self.res_block3_2 = make_layer(
            ResidualBlockNoBN,
            num_blocks[3],
            mid_channels=mid_channels,
            res_scale=res_scale)
        self.res_block3_3 = make_layer(
            ResidualBlockNoBN,
            num_blocks[3],
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last3_1 = _conv3x3_layer(mid_channels, mid_channels)
        self.conv_last3_2 = _conv3x3_layer(mid_channels, mid_channels)
        self.conv_last3_3 = _conv3x3_layer(mid_channels, mid_channels)

        # end, merge features
        self.merge_features = MergeFeatures(mid_channels, out_channels)

    def forward(self, x, soft_attention, textures):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            soft_attention (Tensor): Soft-Attention tensor with shape
                (n, 1, h, w).
            textures (Tuple[Tensor]): Transferred HR texture tensors.
                [(N, C, H, W), (N, C/2, 2H, 2W), ...]

        Returns:
            Tensor: Forward results.
        """

        assert textures[-1].shape[1] == self.texture_channels

        x1 = self.sfe(x)

        # stage 1
        x1_res = torch.cat((x1, textures[0]), dim=1)
        x1_res = self.conv_first1(x1_res)

        # soft-attention
        x1 = x1 + x1_res * soft_attention

        x1_res = self.res_block1(x1)
        x1_res = self.conv_last1(x1_res)

        x1 = x1 + x1_res

        # stage 2
        x21 = x1
        x22 = self.up1(x1)
        x22 = F.relu(x22)

        x22_res = torch.cat((x22, textures[1]), dim=1)
        x22_res = self.conv_first2(x22_res)

        # soft-attention
        x22_res = x22_res * F.interpolate(
            soft_attention,
            scale_factor=2,
            mode='bicubic',
            align_corners=False)
        x22 = x22 + x22_res

        x21_res, x22_res = self.csfi2(x21, x22)

        x21_res = self.res_block2_1(x21_res)
        x22_res = self.res_block2_2(x22_res)

        x21_res = self.conv_last2_1(x21_res)
        x22_res = self.conv_last2_2(x22_res)

        x21 = x21 + x21_res
        x22 = x22 + x22_res

        # stage 3
        x31 = x21
        x32 = x22
        x33 = self.up2(x22)
        x33 = F.relu(x33)

        x33_res = torch.cat((x33, textures[2]), dim=1)
        x33_res = self.conv_first3(x33_res)

        # soft-attention
        x33_res = x33_res * F.interpolate(
            soft_attention,
            scale_factor=4,
            mode='bicubic',
            align_corners=False)
        x33 = x33 + x33_res

        x31_res, x32_res, x33_res = self.csfi3(x31, x32, x33)

        x31_res = self.res_block3_1(x31_res)
        x32_res = self.res_block3_2(x32_res)
        x33_res = self.res_block3_3(x33_res)

        x31_res = self.conv_last3_1(x31_res)
        x32_res = self.conv_last3_2(x32_res)
        x33_res = self.conv_last3_3(x33_res)

        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res
        x = self.merge_features(x31, x32, x33)

        return x

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
