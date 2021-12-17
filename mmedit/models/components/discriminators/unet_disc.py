# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint
from torch.nn.utils import spectral_norm

from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class UNetDiscriminatorWithSpectralNorm(nn.Module):
    """A U-Net discriminator with spectral normalization.

    Args:
        in_channels (int): Channel number of the input.
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, in_channels, mid_channels=64, skip_connection=True):

        super().__init__()

        self.skip_connection = skip_connection

        self.conv_0 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        # downsample
        self.conv_1 = spectral_norm(
            nn.Conv2d(mid_channels, mid_channels * 2, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(
            nn.Conv2d(mid_channels * 2, mid_channels * 4, 4, 2, 1, bias=False))
        self.conv_3 = spectral_norm(
            nn.Conv2d(mid_channels * 4, mid_channels * 8, 4, 2, 1, bias=False))

        # upsample
        self.conv_4 = spectral_norm(
            nn.Conv2d(mid_channels * 8, mid_channels * 4, 3, 1, 1, bias=False))
        self.conv_5 = spectral_norm(
            nn.Conv2d(mid_channels * 4, mid_channels * 2, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=False))

        # final layers
        self.conv_7 = spectral_norm(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(mid_channels, 1, 3, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        """Forward function.

        Args:
            img (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat_0 = self.lrelu(self.conv_0(img))

        # downsample
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_3 = self.lrelu(self.conv_3(feat_2))

        # upsample
        feat_3 = self.upsample(feat_3)
        feat_4 = self.lrelu(self.conv_4(feat_3))
        if self.skip_connection:
            feat_4 = feat_4 + feat_2

        feat_4 = self.upsample(feat_4)
        feat_5 = self.lrelu(self.conv_5(feat_4))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1

        feat_5 = self.upsample(feat_5)
        feat_6 = self.lrelu(self.conv_6(feat_5))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0

        # final layers
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))

        return self.conv_9(out)

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
        elif pretrained is not None:  # Use PyTorch default initialization.
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
