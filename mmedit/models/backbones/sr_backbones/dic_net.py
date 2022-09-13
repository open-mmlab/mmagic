# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.common import make_layer
from mmedit.models.extractors import FeedbackHourglass, reduce_to_five_heatmaps
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class FeedbackBlock(nn.Module):
    """Feedback Block of DIC.

    It has a style of:

    ::
        ----- Module ----->
          ^            |
          |____________|

    Args:
        mid_channels (int): Number of channels in the intermediate features.
        num_blocks (int): Number of blocks.
        upscale_factor (int): upscale factor.
    """

    def __init__(self,
                 mid_channels,
                 num_blocks,
                 upscale_factor,
                 padding=2,
                 prelu_init=0.2):
        super().__init__()

        stride = upscale_factor
        kernel_size = upscale_factor + 4

        self.num_blocks = num_blocks
        self.need_reset = True
        self.last_hidden = None

        self.conv_first = nn.Sequential(
            nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1),
            nn.PReLU(init=prelu_init))

        self.up_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.lr_blocks = nn.ModuleList()
        self.hr_blocks = nn.ModuleList()

        for idx in range(self.num_blocks):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size,
                                       stride, padding),
                    nn.PReLU(init=prelu_init)))
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, kernel_size, stride,
                              padding), nn.PReLU(init=prelu_init)))
            if idx > 0:
                self.lr_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            mid_channels * (idx + 1),
                            mid_channels,
                            kernel_size=1), nn.PReLU(init=prelu_init)))
                self.hr_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            mid_channels * (idx + 1),
                            mid_channels,
                            kernel_size=1), nn.PReLU(init=prelu_init)))

        self.conv_last = nn.Sequential(
            nn.Conv2d(num_blocks * mid_channels, mid_channels, kernel_size=1),
            nn.PReLU(init=prelu_init))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.need_reset:
            self.last_hidden = x
            self.need_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.conv_first(x)

        lr_features = [x]
        hr_features = []

        for idx in range(self.num_blocks):
            # when idx == 0, lr_features == [x]
            lr = torch.cat(lr_features, 1)
            if idx > 0:
                lr = self.lr_blocks[idx - 1](lr)
            hr = self.up_blocks[idx](lr)

            hr_features.append(hr)

            hr = torch.cat(hr_features, 1)
            if idx > 0:
                hr = self.hr_blocks[idx - 1](hr)
            lr = self.down_blocks[idx](hr)

            lr_features.append(lr)

        output = torch.cat(lr_features[1:], 1)
        output = self.conv_last(output)

        self.last_hidden = output

        return output


class FeedbackBlockCustom(FeedbackBlock):
    """Custom feedback block, will be used as the first feedback block.

    Args:
        in_channels (int): Number of channels in the input features.
        mid_channels (int): Number of channels in the intermediate features.
        num_blocks (int): Number of blocks.
        upscale_factor (int): upscale factor.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, upscale_factor):
        super().__init__(mid_channels, num_blocks, upscale_factor)

        prelu_init = 0.2
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.PReLU(init=prelu_init))

    def forward(self, x):
        x = self.conv_first(x)

        lr_features = [x]
        hr_features = []

        for idx in range(self.num_blocks):
            # when idx == 0, lr_features == [x]
            lr = torch.cat(lr_features, 1)
            if idx > 0:
                lr = self.lr_blocks[idx - 1](lr)
            hr = self.up_blocks[idx](lr)

            hr_features.append(hr)

            hr = torch.cat(hr_features, 1)
            if idx > 0:
                hr = self.hr_blocks[idx - 1](hr)
            lr = self.down_blocks[idx](hr)

            lr_features.append(lr)

        output = torch.cat(lr_features[1:], 1)
        output = self.conv_last(output)

        return output


class GroupResBlock(nn.Module):
    """ResBlock with Group Conv.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        mid_channels (int): Channel number of intermediate features.
        groups (int): Number of blocked connections from input to output.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 groups,
                 res_scale=1.0):
        super().__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, groups=groups),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, groups=groups))
        self.res_scale = res_scale

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        res = self.res(x).mul(self.res_scale)
        return x + res


class FeatureHeatmapFusingBlock(nn.Module):
    """Fusing Feature and Heatmap.

    Args:
        in_channels (int): Number of channels in the input features.
        num_heatmaps (int): Number of heatmap.
        num_blocks (int): Number of blocks.
        mid_channels (int | None): Number of channels in the intermediate
            features. Default: None
    """

    def __init__(self,
                 in_channels,
                 num_heatmaps,
                 num_blocks,
                 mid_channels=None):
        super().__init__()

        self.num_heatmaps = num_heatmaps
        res_block_channel = in_channels * num_heatmaps
        if mid_channels is None:
            self.mid_channels = num_heatmaps * in_channels
        else:
            self.mid_channels = mid_channels
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, res_block_channel, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body = make_layer(
            GroupResBlock,
            num_blocks,
            in_channels=res_block_channel,
            out_channels=res_block_channel,
            mid_channels=self.mid_channels,
            groups=num_heatmaps)

    def forward(self, feature, heatmap):
        """Forward function.

        Args:
            feature (Tensor): Input feature tensor.
            heatmap (Tensor): Input heatmap tensor.

        Returns:
            Tensor: Forward results.
        """

        assert self.num_heatmaps == heatmap.size(1)
        batch_size = heatmap.size(0)
        w, h = feature.shape[-2:]

        feature = self.conv_first(feature)
        # B * (num_heatmaps*in_channels) * h * w
        feature = self.body(feature)
        attention = nn.functional.softmax(
            heatmap, dim=1)  # B * num_heatmaps * h * w

        feature = feature.view(batch_size, self.num_heatmaps, -1, w,
                               h) * attention.unsqueeze(2)
        feature = feature.sum(1)
        return feature


class FeedbackBlockHeatmapAttention(FeedbackBlock):
    """Feedback block with HeatmapAttention.

    Args:
        in_channels (int): Number of channels in the input features.
        mid_channels (int): Number of channels in the intermediate features.
        num_blocks (int): Number of blocks.
        upscale_factor (int): upscale factor.
    """

    def __init__(self,
                 mid_channels,
                 num_blocks,
                 upscale_factor,
                 num_heatmaps,
                 num_fusion_blocks,
                 padding=2,
                 prelu_init=0.2):

        super().__init__(
            mid_channels,
            num_blocks,
            upscale_factor,
            padding=padding,
            prelu_init=prelu_init)
        self.fusion_block = FeatureHeatmapFusingBlock(mid_channels,
                                                      num_heatmaps,
                                                      num_fusion_blocks)

    def forward(self, x, heatmap):
        """Forward function.

        Args:
            x (Tensor): Input feature tensor.
            heatmap (Tensor): Input heatmap tensor.

        Returns:
            Tensor: Forward results.
        """

        if self.need_reset:
            self.last_hidden = x
            self.need_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.conv_first(x)

        # fusion
        x = self.fusion_block(x, heatmap)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_blocks):
            # when idx == 0, lr_features == [x]
            lr = torch.cat(lr_features, 1)
            if idx > 0:
                lr = self.lr_blocks[idx - 1](lr)
            hr = self.up_blocks[idx](lr)

            hr_features.append(hr)

            hr = torch.cat(hr_features, 1)
            if idx > 0:
                hr = self.hr_blocks[idx - 1](hr)
            lr = self.down_blocks[idx](hr)

            lr_features.append(lr)

        output = torch.cat(lr_features[1:], 1)
        output = self.conv_last(output)

        self.last_hidden = output

        return output


@BACKBONES.register_module()
class DICNet(nn.Module):
    """DIC network structure for face super-resolution.

    Paper: Deep Face Super-Resolution with Iterative Collaboration between
        Attentive Recovery and Landmark Estimation

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels in the output image
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (tuple[int]): Block numbers in the trunk network.
            Default: 6
        hg_mid_channels (int): Channel number of intermediate features
            of HourGlass. Default: 256
        hg_num_keypoints (int): Keypoint number of HourGlass. Default: 68
        num_steps (int): Number of iterative steps. Default: 4
        upscale_factor (int): Upsampling factor. Default: 8
        detach_attention (bool): Detached from the current tensor for heatmap
            or not.
        prelu_init (float): `init` of PReLU. Default: 0.2
        num_heatmaps (int): Number of heatmaps. Default: 5
        num_fusion_blocks (int): Number of fusion blocks. Default: 7
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 num_blocks=6,
                 hg_mid_channels=256,
                 hg_num_keypoints=68,
                 num_steps=4,
                 upscale_factor=8,
                 detach_attention=False,
                 prelu_init=0.2,
                 num_heatmaps=5,
                 num_fusion_blocks=7):

        super().__init__()

        self.num_steps = num_steps
        self.detach_attention = detach_attention

        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * 4, 3, 1, 1),
            nn.PReLU(init=prelu_init), nn.PixelShuffle(2))

        self.first_block = FeedbackBlockCustom(
            in_channels=mid_channels,
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            upscale_factor=upscale_factor)

        self.block = FeedbackBlockHeatmapAttention(
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            upscale_factor=upscale_factor,
            num_heatmaps=num_heatmaps,
            num_fusion_blocks=num_fusion_blocks)
        self.block.need_reset = False

        self.hour_glass = FeedbackHourglass(
            mid_channels=hg_mid_channels, num_keypoints=hg_num_keypoints)

        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 8, 4, 2),
            nn.PReLU(init=prelu_init),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Forward results.
            sr_outputs (list[Tensor]): forward sr results.
            heatmap_outputs (list[Tensor]): forward heatmap results.
        """

        inter_res = nn.functional.interpolate(
            x, size=(128, 128), mode='bilinear', align_corners=False)

        x = self.conv_first(x)

        sr_outputs = []
        heatmap_outputs = []
        last_hidden = None
        heatmap = None

        for step in range(self.num_steps):
            if step == 0:
                sr_feature = self.first_block(x)
                self.block.last_hidden = sr_feature
            else:
                heatmap = reduce_to_five_heatmaps(heatmap,
                                                  self.detach_attention)
                sr_feature = self.block(x, heatmap)

            sr = self.conv_last(sr_feature)
            sr = torch.add(inter_res, sr)
            heatmap, last_hidden = self.hour_glass(sr, last_hidden)

            sr_outputs.append(sr)
            heatmap_outputs.append(heatmap)

        return sr_outputs, heatmap_outputs

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
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
