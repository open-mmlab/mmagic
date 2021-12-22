import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmedit.models.common import make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


def pixel_shuffle(x, scale=8, up=True):
    """Up-scale or down-scale by pixel shuffle.

    Args:
        x (Tensor): Input tensor.
        scale (int): Scale factor.
        up (bool): Up-scale or down-scale.

    Returns:
        y (Tensor): Output tensor.
    """

    b, c, h, w = x.shape
    if up:
        y = F.pixel_shuffle(x, scale)
    else:
        h = int(h / scale)
        w = int(w / scale)
        x = x.view(b, c, h, scale, w, scale)
        shuffle_out = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        y = shuffle_out.view(b, c * scale * scale, h, w)
    return y


def get_padding_functions(x, padding=7):
    """Generate padding function.

    tensor --padding_input--> padded tensor
       ↑                            |
       ------padding_output----------

    Args:
        x (Tensor): Input tensor.
        padding (int): Padding size.

    Returns:
        padding_input (Function): Padding function.
        padding_output (Function): Depadding function.
    """

    h, w = x.shape[-2:]
    padding_width, padding_height = 0, 0
    if w != ((w >> padding) << padding):
        padding_width = (((w >> padding) + 1) << padding) - w
    if h != ((h >> padding) << padding):
        padding_height = (((h >> padding) + 1) << padding) - h
    left, right = padding_width // 2, padding_width - padding_width // 2
    up, down = padding_height // 2, padding_height - padding_height // 2
    # print(up, down, left, right)
    if down >= h or right >= w:
        function = nn.ReplicationPad2d
    else:
        function = nn.ReflectionPad2d
    padding_input = function(padding=[left, right, up, down])
    padding_output = function(padding=[0 - left, 0 - right, 0 - up, 0 - down])
    return padding_input, padding_output


class ConvNorm(nn.Module):
    """ReflectionPad2d, Conv2d (and Norm).

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of input features.
        kernel_size (int): Kernel size of convolution layer.
        stride (int): Stride of size of convolution layer. Default: 1.
        norm (None | function): Norm layer. If None, no norm layer.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 norm=None):
        super().__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=kernel_size,
            bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(
                out_channels, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Forward function for ConvNorm.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor with shape (n, c, h, w).
        """

        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class CALayer(nn.Module):
    """Channel Attention Layer.

    Args:
        mid_channels (int): Channel number of the intermediate features.
        reduction (int): Channel reduction of CA. Default: 16.
    """

    def __init__(self, mid_channels, reduction=16):
        super().__init__()

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.channel_attention = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels // reduction,
                1,
                padding=0,
                bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels // reduction,
                mid_channels,
                1,
                padding=0,
                bias=True), nn.Sigmoid())

    def forward(self, x):
        """Forward function for CALayer.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor with shape (n, c, h, w).
            Tensor: CA tensor with shape (n, c, 1, 1).
        """

        y = self.avg_pool(x)
        y = self.channel_attention(y)
        return x * y, y


class RCA(nn.Module):
    """Residual Channel Attention Module.

    Args:
        mid_channels (int): Channel number of the intermediate features.
        kernel_size (int): Kernel size of RCA. Default: 3.
        downscale (bool): Down scale or not. Default: False. Default: False.
        reduction (int): Channel reduction of CA. Default: 16.
        return_ca (bool): Return CA tensor or not. Default: False.
        norm (None | function): Norm layer. If None, no norm layer.
            Default: None.
        act (function): activate function. Default: nn.ReLU(True).
    """

    def __init__(self,
                 mid_channels,
                 kernel_size=3,
                 downscale=False,
                 reduction=16,
                 return_ca=False,
                 norm=False,
                 act=nn.ReLU(True)):
        super().__init__()

        self.body = nn.Sequential(
            ConvNorm(
                mid_channels,
                mid_channels,
                kernel_size,
                stride=2 if downscale else 1,
                norm=norm), act,
            ConvNorm(
                mid_channels, mid_channels, kernel_size, stride=1, norm=norm),
            CALayer(mid_channels, reduction))
        self.return_ca = return_ca
        self.down_scale = downscale
        if self.down_scale:
            self.down_conv = nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Forward function for RCA.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            if return_ca:
                Tensor: Output tensor with shape (n, c, h, w).
                Tensor: CA tensor with shape (n, c, 1, 1).
            else:
                Tensor: Output tensor with shape (n, c, h, w).
        """

        res = x
        out, ca = self.body(x)

        if self.down_scale:
            res = self.down_conv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out


class ResidualGroup(nn.Module):
    """Residual Group.

    Args:
        block_layer (nn.module): nn.module class for basic block.
        num_block_layers (int): number of blocks.
        mid_channels (int): Channel number of the intermediate features.
        kernel_size (int): Kernel size of ResidualGroup.
        reduction (int): Channel reduction of CA. Default: 16.
        act (function): activate function. Default: nn.ReLU(True).
        norm (None | function): Norm layer. If None, no norm layer.
            Default: None.
    """

    def __init__(self,
                 block_layer,
                 num_block_layers,
                 mid_channels,
                 kernel_size,
                 reduction,
                 act,
                 norm=False):
        super().__init__()

        self.body = make_layer(
            block_layer,
            num_block_layers,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            reduction=reduction,
            norm=norm,
            act=act)
        self.conv_after_body = ConvNorm(
            mid_channels, mid_channels, kernel_size, stride=1, norm=norm)

    def forward(self, x):
        """Forward function for ResidualGroup.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor with shape (n, c, h, w).
        """

        y = self.body(x)
        y = self.conv_after_body(y)
        return x + y


@BACKBONES.register_module()
class CAINNet(nn.Module):
    """CAIN network structure.

    Paper: Channel Attention Is All You Need for Video Frame Interpolation.
    Ref repo: https://github.com/myungsub/CAIN

    Args:
        in_channels (int): Channel number of inputs. Default: 3.
        out_channels (int): Channel number of outputs. Default: 3.
        kernel_size (int): Kernel size of CAINNet. Default: 3.
        num_block_groups (int): Number of block groups. Default: 5.
        num_block_layers (int): Number of blocks in a group. Default: 12.
        depth (int): Down scale depth, scale = 2**depth. Default: 3.
        reduction (int): Channel reduction of CA. Default: 16.
        norm (None | function): Norm layer. If None, no norm layer.
            Default: None.
        act (function): activate function. Default: nn.LeakyReLU(0.2, True)).
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 kernel_size=3,
                 num_block_groups=5,
                 num_block_layers=12,
                 depth=3,
                 reduction=16,
                 padding=7,
                 act=nn.LeakyReLU(0.2, True)):
        super().__init__()

        mid_channels = in_channels * pow(4, depth)
        self.scale = pow(2, depth)
        self.padding = padding

        self.conv_first = nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_block_groups,
            block_layer=RCA,
            num_block_layers=num_block_layers,
            mid_channels=mid_channels,
            kernel_size=3,
            reduction=reduction,
            act=act)
        self.conv_last = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, xs, padding_flag=False):
        """Forward function.

        Args:
            xs (Tensor): Input tensor with shape (n, 2, c, h, w).
            padding_flag (bool): Padding or not. Default: False.

        Returns:
            Tensor: Forward results.
        """

        assert xs.shape[1] == 2
        x1, x2 = xs[:, 0], xs[:, 1]

        mean1 = x1.mean(2, keepdim=True).mean(3, keepdim=True)
        mean2 = x2.mean(2, keepdim=True).mean(3, keepdim=True)
        x1 -= mean1
        x2 -= mean2

        if padding_flag:
            padding_input, padding_output = get_padding_functions(
                x1, self.padding)
            x1 = padding_input(x1)
            x2 = padding_input(x2)

        x1 = pixel_shuffle(x1, scale=self.scale, up=False)
        x2 = pixel_shuffle(x2, scale=self.scale, up=False)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv_first(x)
        res = self.body(x)
        res += x
        x = self.conv_last(res)
        x = pixel_shuffle(x, scale=self.scale, up=True)

        if padding_flag:
            x = padding_output(x)

        x += (mean1 + mean2) / 2
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
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
