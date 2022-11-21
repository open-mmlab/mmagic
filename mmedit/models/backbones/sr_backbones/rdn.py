import torch
from mmcv.runner import load_checkpoint
from torch import nn

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class DenseLayer(nn.Module):
    """Dense layer.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c_in, h, w).

        Returns:
            Tensor: Forward results, tensor with shape (n, c_in+c_out, h, w).
        """
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    """Residual Dense Block of Residual Dense Network.

    Args:
        in_channels (int): Channel number of inputs.
        channel_growth (int): Channels growth in each layer.
        num_layers (int): Layer number in the Residual Dense Block.
    """

    def __init__(self, in_channels, channel_growth, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[
            DenseLayer(in_channels + channel_growth * i, channel_growth)
            for i in range(num_layers)
        ])

        # local feature fusion
        self.lff = nn.Conv2d(
            in_channels + channel_growth * num_layers,
            in_channels,
            kernel_size=1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return x + self.lff(self.layers(x))  # local residual learning


@BACKBONES.register_module()
class RDN(nn.Module):
    """RDN model for single image super-resolution.

    Paper: Residual Dense Network for Image Super-Resolution

    Adapted from 'https://github.com/yjn870/RDN-pytorch.git'
    'RDN-pytorch/blob/master/models.py'
    Copyright (c) 2021, JaeYun Yeo, under MIT License.

    Most of the implementation follows the implementation in:
    'https://github.com/sanghyun-son/EDSR-PyTorch.git'
    'EDSR-PyTorch/blob/master/src/model/rdn.py'
    Copyright (c) 2017, sanghyun-son, under MIT license.


    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        num_layer (int): Layer number in the Residual Dense Block.
            Default: 8.
        channel_growth(int): Channels growth in each layer of RDB.
            Default: 64.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=16,
                 upscale_factor=4,
                 num_layers=8,
                 channel_growth=64):

        super().__init__()
        self.mid_channels = mid_channels
        self.channel_growth = channel_growth
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.rdbs.append(
                RDB(self.mid_channels, self.channel_growth, self.num_layers))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(
                self.mid_channels * self.num_blocks,
                self.mid_channels,
                kernel_size=1),
            nn.Conv2d(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                padding=3 // 2))

        # up-sampling
        assert 2 <= upscale_factor <= 4
        if upscale_factor == 2 or upscale_factor == 4:
            self.upscale = []
            for _ in range(upscale_factor // 2):
                self.upscale.extend([
                    nn.Conv2d(
                        self.mid_channels,
                        self.mid_channels * (2**2),
                        kernel_size=3,
                        padding=3 // 2),
                    nn.PixelShuffle(2)
                ])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    self.mid_channels,
                    self.mid_channels * (upscale_factor**2),
                    kernel_size=3,
                    padding=3 // 2), nn.PixelShuffle(upscale_factor))

        self.output = nn.Conv2d(
            self.mid_channels, out_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1
        # global residual learning
        x = self.upscale(x)
        x = self.output(x)
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
