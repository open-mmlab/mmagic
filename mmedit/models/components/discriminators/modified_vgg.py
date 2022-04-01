# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class ModifiedVGG(nn.Module):
    """A modified VGG discriminator with input size 128 x 128.

    It is used to train SRGAN and ESRGAN.

    Args:
        in_channels (int): Channel number of inputs. Default: 3.
        mid_channels (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.conv0_0 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(
            mid_channels, mid_channels, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(mid_channels, affine=True)

        self.conv1_0 = nn.Conv2d(
            mid_channels, mid_channels * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(mid_channels * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            mid_channels * 2, mid_channels * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(mid_channels * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            mid_channels * 2, mid_channels * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(mid_channels * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            mid_channels * 4, mid_channels * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(mid_channels * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            mid_channels * 4, mid_channels * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(mid_channels * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(mid_channels * 8, affine=True)

        self.conv4_0 = nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(mid_channels * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(mid_channels * 8, affine=True)

        self.linear1 = nn.Linear(mid_channels * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        assert x.size(2) == 128 and x.size(3) == 128, (
            f'Input spatial size must be 128x128, '
            f'but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

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
            pass  # Use PyTorch default initialization.
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
