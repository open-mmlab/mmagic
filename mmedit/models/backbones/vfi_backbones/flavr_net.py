# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class FLAVRNet(nn.Module):
    """PyTorch implementation of FLAVR for video frame interpolation.

    Paper:
        FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
    Ref repo: https://github.com/tarun005/FLAVR

    Args:
        num_input_frames (int): Number of input frames.
        num_output_frames (int): Number of output frames.
        mid_channels_list (list[int]): List of number of mid channels.
            Default: [512, 256, 128, 64]
        encoder_layers_list (list[int]): List of number of layers in encoder.
            Default: [2, 2, 2, 2]
        bias (bool): If ``True``, adds a learnable bias to the conv layers.
            Default: ``True``
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: None
        join_type (str): Join type of tensors from decoder and encoder.
            Candidates are ``concat`` and ``add``. Default: ``concat``
        up_mode (str): Up-mode UpConv3d, candidates are ``transpose`` and
            ``trilinear``. Default: ``transpose``
    """

    def __init__(self,
                 num_input_frames,
                 num_output_frames,
                 mid_channels_list=[512, 256, 128, 64],
                 encoder_layers_list=[2, 2, 2, 2],
                 bias=False,
                 norm_cfg=None,
                 join_type='concat',
                 up_mode='transpose'):
        super().__init__()

        self.encoder = Encoder(
            block=BasicBlock,
            layers=encoder_layers_list,
            stem_layer=BasicStem,
            mid_channels_list=mid_channels_list[::-1],
            bias=bias,
            norm_cfg=norm_cfg)

        self.decoder = Decoder(
            join_type=join_type,
            up_mode=up_mode,
            mid_channels_list=mid_channels_list,
            batchnorm=norm_cfg)

        self.feature_fuse = ConvModule(
            mid_channels_list[3] * num_input_frames,
            mid_channels_list[3],
            kernel_size=1,
            stride=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2, inplace=True))

        out_channels = 3 * num_output_frames
        self.conv_last = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                mid_channels_list[3],
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=0))

    def forward(self, images: torch.Tensor):
        # from [b, t, c, h, w] to [b, c, d, h, w], where t==d
        images = images.permute((0, 2, 1, 3, 4))

        # Batch mean normalization works slightly better than global mean
        # normalization, Refer to https://github.com/myungsub/CAIN
        mean_ = images.mean((2, 3, 4), keepdim=True)
        images = images - mean_

        xs = self.encoder(images)

        dx_out = self.decoder(xs)

        out = self.feature_fuse(dx_out)
        out = self.conv_last(out)
        # b, t*c, h, w

        b, c_all, h, w = out.shape
        t = c_all // 3
        mean_ = mean_.view(b, 1, 3, 1, 1)
        out = out.view(b, t, 3, h, w)
        out = out + mean_

        # if t==1, which means the output only contains one frame.
        out = out.squeeze(1)

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
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class Encoder(nn.Module):
    """Encoder of FLAVR.

    Args:
        block (nn.Module): Basic block of encoder.
        layers (str): List of layers in encoder.
        stem_layer (nn.Module): stem layer (conv first).
        mid_channels_list (list[int]): List of mid channels.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: None
        bias (bool): If ``True``, adds a learnable bias to the conv layers.
            Default: ``True``
    """

    def __init__(self, block, layers, stem_layer, mid_channels_list, norm_cfg,
                 bias):
        super().__init__()

        self.in_channels = mid_channels_list[0]
        self.bias = bias

        self.stem_layer = stem_layer(mid_channels_list[0], bias, norm_cfg)

        self.layer1 = self._make_layer(
            block,
            mid_channels_list[0],
            layers[0],
            norm_cfg=norm_cfg,
            stride=1)
        self.layer2 = self._make_layer(
            block,
            mid_channels_list[1],
            layers[1],
            norm_cfg=norm_cfg,
            stride=2,
            temporal_stride=1)
        self.layer3 = self._make_layer(
            block,
            mid_channels_list[2],
            layers[2],
            norm_cfg=norm_cfg,
            stride=2,
            temporal_stride=1)
        self.layer4 = self._make_layer(
            block,
            mid_channels_list[3],
            layers[3],
            norm_cfg=norm_cfg,
            stride=1,
            temporal_stride=1)

        # init weights
        self._initialize_weights()

    def forward(self, x):

        x_0 = self.stem_layer(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        return x_0, x_1, x_2, x_3, x_4

    def _make_layer(self,
                    block,
                    mid_channels,
                    num_blocks,
                    norm_cfg,
                    stride=1,
                    temporal_stride=None):
        downsample = None

        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            if temporal_stride:
                ds_stride = (temporal_stride, stride, stride)
            else:
                ds_stride = (stride, stride, stride)
            downsample = ConvModule(
                self.in_channels,
                mid_channels * block.expansion,
                kernel_size=1,
                stride=ds_stride,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=None)
            stride = ds_stride

        layers = []
        layers.append(
            block(
                self.in_channels,
                mid_channels,
                norm_cfg=norm_cfg,
                stride=stride,
                bias=self.bias,
                downsample=downsample))

        self.in_channels = mid_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.in_channels,
                    mid_channels,
                    norm_cfg=norm_cfg,
                    bias=self.bias))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    """Decoder of FLAVR.

    Args:
        join_type (str): Join type of tensors from decoder and encoder.
            Candidates are ``concat`` and ``add``. Default: ``concat``
        up_mode (str): Up-mode UpConv3d, candidates are ``transpose`` and
            ``trilinear``. Default: ``transpose``
        mid_channels_list (list[int]): List of mid channels.
            Default: [512, 256, 128, 64]
        batchnorm (bool): Whether contains BatchNorm3d. Default: False.
    """

    def __init__(self,
                 join_type,
                 up_mode,
                 mid_channels_list=[512, 256, 128, 64],
                 batchnorm=False):
        super().__init__()

        growth = 2 if join_type == 'concat' else 1
        self.join_type = join_type
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.layer0 = Conv3d(
            mid_channels_list[0],
            mid_channels_list[1],
            kernel_size=3,
            padding=1,
            bias=True,
            batchnorm=batchnorm)
        self.layer1 = UpConv3d(
            mid_channels_list[1] * growth,
            mid_channels_list[2],
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            up_mode=up_mode,
            batchnorm=batchnorm)
        self.layer2 = UpConv3d(
            mid_channels_list[2] * growth,
            mid_channels_list[3],
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            up_mode=up_mode,
            batchnorm=batchnorm)
        self.layer3 = Conv3d(
            mid_channels_list[3] * growth,
            mid_channels_list[3],
            kernel_size=3,
            padding=1,
            bias=True,
            batchnorm=batchnorm)
        self.layer4 = UpConv3d(
            mid_channels_list[3] * growth,
            mid_channels_list[3],
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            up_mode=up_mode,
            batchnorm=batchnorm)

    def forward(self, xs):

        dx_3 = self.lrelu(self.layer0(xs[4]))
        dx_3 = self._join_tensors(dx_3, xs[3])

        dx_2 = self.lrelu(self.layer1(dx_3))
        dx_2 = self._join_tensors(dx_2, xs[2])

        dx_1 = self.lrelu(self.layer2(dx_2))
        dx_1 = self._join_tensors(dx_1, xs[1])

        dx_0 = self.lrelu(self.layer3(dx_1))
        dx_0 = self._join_tensors(dx_0, xs[0])

        dx_out = self.lrelu(self.layer4(dx_0))
        dx_out = torch.cat(torch.unbind(dx_out, 2), 1)

        return dx_out

    def _join_tensors(self, x1, x2):
        """Concat or Add two tensors.

        Args:
            x1 (Tensor): The first input tensor.
            x2 (Tensor): The second input tensor.
        """

        if self.join_type == 'concat':
            return torch.cat([x1, x2], dim=1)
        else:
            return x1 + x2


class UpConv3d(nn.Module):
    """A conv block that bundles conv/SEGating/norm layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        up_mode (str): Up-mode UpConv3d, candidates are ``transpose`` and
            ``trilinear``. Default: ``transpose``.
        batchnorm (bool): Whether contains BatchNorm3d. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 up_mode='transpose',
                 batchnorm=False):

        super().__init__()

        self.up_mode = up_mode

        if self.up_mode == 'transpose':
            self.upconv = nn.ModuleList([
                nn.ConvTranspose3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding),
                SEGating(out_channels)
            ])

        else:
            self.upconv = nn.ModuleList([
                nn.Upsample(
                    mode='trilinear',
                    scale_factor=(1, 2, 2),
                    align_corners=False),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                SEGating(out_channels)
            ])

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_channels)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)


class Conv3d(nn.Module):
    """A conv block that bundles conv/SEGating/norm layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        bias (bool): If ``True``, adds a learnable bias to the conv layer.
            Default: ``True``
        batchnorm (bool): Whether contains BatchNorm3d. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 batchnorm=False):

        super().__init__()
        self.conv = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias),
            SEGating(out_channels)
        ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_channels)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)


class BasicStem(ConvModule):
    """The default conv-batchnorm-relu stem of FLAVR.

    Args:
        out_channels (int): Number of output channels. Default: 64
        bias (bool): If ``True``, adds a learnable bias to the conv layer.
            Default: ``False``
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: None.
    """

    def __init__(self, out_channels=64, bias=False, norm_cfg=None):
        super().__init__(
            3,
            out_channels,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=bias,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            inplace=False)


class BasicBlock(nn.Module):
    """Basic block of encoder in FLAVR.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the block.
        stride (int | tuple[int]): Stride of the first convolution.
            Default: 1.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: None.
        bias (bool): If ``True``, adds a learnable bias to the conv layers.
            Default: ``True``
        batchnorm (bool): Whether contains BatchNorm3d. Default: False.
        downsample (None | torch.nn.Module): Down-sample layer.
            Default: None.
    """

    expansion = 1

    def __init__(
        self,
        in_channels,
        mid_channels,
        stride=1,
        norm_cfg=None,
        bias=False,
        downsample=None,
    ):
        super().__init__()

        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 1),
            bias=bias,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=bias,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.fg = SEGating(mid_channels)  # Feature Gating
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEGating(nn.Module):
    """Gatting of SE attention.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, in_channels):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, in_channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):

        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y
