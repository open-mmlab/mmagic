# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.fused_bias_leakyrelu import FusedBiasLeakyReLU, fused_bias_leakyrelu
from mmcv.ops.upfirdn2d import upfirdn2d
from mmedit.registry import MODULES

from ..pggan.pggan_modules import EqualizedLRLinearModule
from ..stylegan1 import make_kernel
from ..stylegan2 import UpsampleUpFIRDn, ModulatedConv2d, ModulatedToRGB


class _FusedBiasLeakyReLU(FusedBiasLeakyReLU):
    """Wrap FusedBiasLeakyReLU to support FP16 training."""

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        return fused_bias_leakyrelu(x, self.bias.to(x.dtype),
                                    self.negative_slope, self.scale)


class NormStyleCode(nn.Module):
    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Deafult: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * (upsample_factor**2)

        if upsample_factor > 1:
            pad = (self.kernel.shape[0] - upsample_factor) - (kernel_size - 1)
            self.pad = ((pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1)
        elif downsample_factor > 1:
            pad = (self.kernel.shape[0] - downsample_factor) + (kernel_size - 1)
            self.pad = ((pad + 1) // 2, pad // 2)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out


class EqualLinear(EqualizedLRLinearModule):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, activation=None):
        super(EqualLinear, self).__init__(
            in_channels,
            out_channels,
            bias=bias,
        )
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f'Wrong activation value in EqualLinear: {activation}'
                             "Supported ones are: ['fused_lrelu', None].")
        self.scale = (1 / math.sqrt(in_channels)) * self.lr_mul

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_bias_leakyrelu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out


class GcfsrModulatedConv2d(ModulatedConv2d):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in GcfsrModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 upsample=False,
                 downsample=True,
                 resample_kernel=[1, 3, 3, 1],
                 eps=1e-8):
        super(GcfsrModulatedConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_channels=num_style_feat,
            demodulate=demodulate,
            upsample=upsample,
            downsample=downsample,
            blur_kernel=resample_kernel,
            equalized_lr_cfg=None,
            eps=eps,
        )

        self.upsample = upsample
        self.downsample = downsample
        assert not(upsample and downsample), f"upsample and downsample should be not set to True at the same time, but get upsample={upsample} and downsample={downsample}"

        if upsample or downsample:
            upsample_factor, downsample_factor = 1, 1
            if upsample and not downsample:
                upsample_factor = 2
            elif downsample and not upsample:
                downsample_factor = 2
            self.smooth = UpFirDnSmooth(
                resample_kernel, 
                upsample_factor=upsample_factor, 
                downsample_factor=downsample_factor, 
                kernel_size=kernel_size
            )

        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        # modulation inside each modulated conv
        self.modulation = EqualLinear(
            num_style_feat, 
            in_channels, 
            bias=True, 
            activation=None
        )

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.scale * self.weight * style  # (b, c_out, c_in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.upsample:
            x = x.view(1, b * c, h, w)
            weight = weight.view(b, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(b * c, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
            out = self.smooth(out)
        elif self.downsample:
            x = self.smooth(x)
            x = x.view(1, b * c, *x.shape[2:4])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        elif self.upsample or not self.downsample:
            x = x.view(1, b * c, h, w)
            # weight: (b*c_out, c_in, k, k), groups=b
            out = F.conv2d(x, weight, padding=self.padding, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        else:
            raise ValueError(f"upsample and downsample should be not set to True at the same time.")

        return out


class StyleConv(nn.Module):
    """Style conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 upsample=False,
                 downsample=True,
                 resample_kernel=[1, 3, 3, 1]):
        super(StyleConv, self).__init__()
        self.conv = GcfsrModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            upsample=upsample,
            downsample=downsample,
            resample_kernel=resample_kernel
        )
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = _FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.conv(x, style)
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # activation (with bias)
        out = self.activate(out)
        return out


class ToRGB(ModulatedToRGB):
    """To RGB from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, num_style_feat, upsample=True, resample_kernel=(1, 3, 3, 1)):
        super(ToRGB, self).__init__(
            in_channels=in_channels,
            style_channels=num_style_feat,
            upsample=upsample,
            blur_kernel=resample_kernel
        )
        if upsample:
            self.upsample = UpsampleUpFIRDn(resample_kernel, factor=2)
        else:
            self.upsample = None
        self.conv = GcfsrModulatedConv2d(
            in_channels, 3, 
            kernel_size=1, 
            num_style_feat=num_style_feat, 
            demodulate=False, 
            upsample=False, 
            downsample=False
        )


class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 resample_kernel=(1, 3, 3, 1),
                 bias=True,
                 activate=True):
        layers = []
        # downsample
        if downsample:
            layers.append(
                UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        # conv
        layers.append(
            EqualConv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias
                and not activate))
        # activation
        if activate:
            if bias:
                layers.append(_FusedBiasLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super(ConvLayer, self).__init__(*layers)


class Norm2Scale(nn.Module):
    def forward(self, scale1, scale2):
        scales_norm = scale1**2 + scale2**2 + 1e-8
        return scale1 * torch.rsqrt(scales_norm), scale2 * torch.rsqrt(scales_norm)


class StyleConv_norm_scale_shift(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 upsample=False,
                 downsample=True,
                 resample_kernel=(1, 3, 3, 1)):
        super(StyleConv_norm_scale_shift, self).__init__()
        self.modulated_conv = GcfsrModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            upsample=upsample,
            downsample=downsample,
            resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = _FusedBiasLeakyReLU(out_channels)
        self.norm = Norm2Scale()

    def forward(self, x, style, noise=None, scale1=None, scale2=None, shift=None):
        # modulate
        out = self.modulated_conv(x, style)
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise

        scale1, scale2 = self.norm(scale1, scale2)

        out = out * scale1.view(-1, out.size(1), 1, 1) + shift * scale2.view(-1, out.size(1), 1, 1)

        # activation (with bias)
        out = self.activate(out)
        return out


@MODULES.register_module()
class GCFSR(nn.Module):
    def __init__(self,
                 out_size,
                 num_style_feat,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 narrow=1):
        super(GCFSR, self).__init__()

        self.num_style_feat = num_style_feat

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.log_size = int(math.log(out_size, 2))
        self.num_latent = (self.log_size - 2) * 2 - 2

        first_out_size = 2**(int(math.log(out_size, 2)))

        self.encoder_channels = channels

        self.conv_body_first = ConvLayer(3, channels[f'{first_out_size}'], 3, bias=True, activate=True)
        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size-1, 2+1, -1):
            out_channels = channels[f'{2**i}']
            self.conv_body_down.append(ConvLayer(in_channels, out_channels, 3, downsample=True))
            in_channels = out_channels

        # to generate "const 16x16";
        self.final_conv = ConvLayer(channels['16'], channels['16'], 3)

        self.final_down1 = ConvLayer(channels['16'], channels['8'], 3, downsample=True)
        self.final_down2 = ConvLayer(channels['8'], channels['4']//2, 3, downsample=True)
        self.final_linear = EqualLinear(2 * 4 * 512, self.num_style_feat * self.num_latent, bias=True, activation='fused_lrelu')

        self.condition_scale1 = nn.ModuleList()
        self.condition_scale2 = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        
        for i in range(self.log_size, 2+1, -1):
            out_channels = channels[f'{2**i}']
            in_channels = channels[f'{2**i}']

            self.condition_scale1.append(EqualLinear(1, out_channels, bias=True, activation=None))

            self.condition_scale2.append(EqualLinear(1, out_channels, bias=True, activation=None))
           
            self.condition_shift.append(
                ConvLayer(in_channels, out_channels, 3, bias=True, activate=False))

        # stylegan decoder
        self.style_conv1 = StyleConv_norm_scale_shift(
            channels['16'],
            channels['16'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            upsample=False,
            downsample=False,
            resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['16'], num_style_feat, upsample=False, resample_kernel=resample_kernel)
        
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2 - 2) * 2 + 1

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels['16']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3+2, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    upsample=True,
                    downsample=False,
                    resample_kernel=resample_kernel,
                ))
            self.style_convs.append(
                StyleConv_norm_scale_shift(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    upsample=False,
                    downsample=False,
                    resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def forward(self, x, in_size,
                noise=None,
                randomize_noise=True,
                return_latents=False):

        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
            
        # main generation
        feat = self.conv_body_first(x)
        
        scales1, scales2, shifts = [], [], []

        scale1 = self.condition_scale1[0](in_size)
        scales1.append(scale1.clone())
        scale2 = self.condition_scale2[0](in_size)
        scales2.append(scale2.clone())

        shift = self.condition_shift[0](feat)
        shifts.append(shift.clone())

        j = 1
        for i in range(len(self.conv_body_down)):
            feat = self.conv_body_down[i](feat)
            if j < len(self.condition_scale1):
                scale1 = self.condition_scale1[j](in_size)
                scales1.append(scale1.clone())
                scale2 = self.condition_scale2[j](in_size)
                scales2.append(scale2.clone())
                shift = self.condition_shift[j](feat)
                shifts.append(shift.clone())
                j += 1

        scales1 = scales1[::-1]
        scales2 = scales2[::-1]
        shifts = shifts[::-1]

        b = feat.size(0)

        tmp = self.final_down2(self.final_down1(feat))
        latent = self.final_linear(tmp.view(b, -1)).view(-1, self.num_latent, self.num_style_feat)

        out = self.final_conv(feat)
        out = self.style_conv1(out, latent[:, 0], noise=noise[0], scale1=scales1[0], scale2=scales2[0], shift=shifts[0])
        # out = out * scales1[0].view(-1, out.size(1), 1, 1) + shifts[0] * scales2[0].view(-1, out.size(1), 1, 1)
        skip = self.to_rgb1(out, latent[:, 1])


        i = 1
        j = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2, scale1=scales1[j], scale2=scales2[j], shift=shifts[j])
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
            j += 1

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


@MODULES.register_module()
class GCFSR_blind(nn.Module):
    def __init__(self,
                 out_size,
                 num_style_feat,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 narrow=1):
        super(GCFSR_blind, self).__init__()

        self.num_style_feat = num_style_feat

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.log_size = int(math.log(out_size, 2))
        self.num_latent = (self.log_size - 2) * 2 - 2

        first_out_size = 2**(int(math.log(out_size, 2)))

        self.encoder_channels = channels

        self.conv_body_first = ConvLayer(3, channels[f'{first_out_size}'], 3, bias=True, activate=True)
        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size-1, 2+1, -1):
            out_channels = channels[f'{2**i}']
            self.conv_body_down.append(ConvLayer(in_channels, out_channels, 3, downsample=True))
            in_channels = out_channels

        # to generate "const 16x16";
        self.final_conv = ConvLayer(channels['16'], channels['16'], 3)

        self.final_down1 = ConvLayer(channels['16'], channels['8'], 3, downsample=True)
        self.final_down2 = ConvLayer(channels['8'], channels['4'], 3, downsample=True)
        self.final_linear = EqualLinear(4 * 4 * 512, self.num_style_feat * self.num_latent, bias=True, activation='fused_lrelu')

        self.condition_scale1 = nn.ModuleList()
        self.condition_scale2 = nn.ModuleList()
        self.condition_shift = nn.ModuleList()

        for i in range(self.log_size, 2+1, -1):
            out_channels = channels[f'{2**i}']
            in_channels = channels[f'{2**i}']

            self.condition_scale1.append(EqualLinear(1, out_channels, bias=True, activation=None))

            self.condition_scale2.append(EqualLinear(1, out_channels, bias=True, activation=None))
           
            self.condition_shift.append(
                ConvLayer(in_channels, out_channels, 3, bias=True, activate=False))

        # stylegan decoder
        self.style_conv1 = StyleConv_norm_scale_shift(
            channels['16'],
            channels['16'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            upsample=False,
            downsample=False,
            resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['16'], num_style_feat, upsample=False, resample_kernel=resample_kernel)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2 - 2) * 2 + 1

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels['16']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3+2, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    upsample=True,
                    downsample=False,
                    resample_kernel=resample_kernel,
                ))
            self.style_convs.append(
                StyleConv_norm_scale_shift(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    upsample=False,
                    downsample=False,
                    resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def forward(self, x,
                noise=None,
                randomize_noise=True,
                return_latents=False):
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]

        device = x.device
        # fix "in_size" to 1
        in_size = torch.ones(1).to(device)

        # main generation
        feat = self.conv_body_first(x)
        
        scales1, scales2, shifts = [], [], []

        scale1 = self.condition_scale1[0](in_size)
        scales1.append(scale1.clone())
        scale2 = self.condition_scale2[0](in_size)
        scales2.append(scale2.clone())

        shift = self.condition_shift[0](feat)
        shifts.append(shift.clone())

        j = 1
        for i in range(len(self.conv_body_down)):
            feat = self.conv_body_down[i](feat)
            if j < len(self.condition_scale1):
                scale1 = self.condition_scale1[j](in_size)
                scales1.append(scale1.clone())
                scale2 = self.condition_scale2[j](in_size)
                scales2.append(scale2.clone())
                shift = self.condition_shift[j](feat)
                shifts.append(shift.clone())
                j += 1

        scales1 = scales1[::-1]
        scales2 = scales2[::-1]
        shifts = shifts[::-1]

        b = feat.size(0)

        tmp = self.final_down2(self.final_down1(feat))
        latent = self.final_linear(tmp.view(b, -1))
        latent = latent.view(-1, self.num_latent, self.num_style_feat)

        out = self.final_conv(feat)
        out = self.style_conv1(out, latent[:, 0], noise=noise[0], scale1=scales1[0], scale2=scales2[0], shift=shifts[0])
        # out = out * scales1[0].view(-1, out.size(1), 1, 1) + shifts[0] * scales2[0].view(-1, out.size(1), 1, 1)
        skip = self.to_rgb1(out, latent[:, 1])


        i = 1
        j = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2, scale1=scales1[j], scale2=scales2[j], shift=shifts[j])
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
            j += 1

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None
