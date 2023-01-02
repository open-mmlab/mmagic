# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import print_log
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from ..stylegan2 import StyleGAN2Generator
from ..stylegan2.stylegan2_modules import ModulatedStyleConv, ModulatedToRGB
from ..stylegan3.stylegan3_modules import FullyConnectedLayer
from ..stylegan3.stylegan3_ops.ops import bias_act, upfirdn2d
from .gmpi_ops import misc, persistence
from .gmpi_ops.ops import conv2d_resample, fma

FLOATING_EPS = 1e-8

#----------------------------------------------------------------------------


# https://github.com/naoto0804/pytorch-AdaIN/blob/197cdc142ae79693af07be7c8eb7c0b9c90ab204/function.py#L4
def calc_mean_std(feat, eps=FLOATING_EPS):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2]
            ), f'{content_feat.size()}, {style_feat.size()}'
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


@misc.profiled_function
def get_embedder(multires, input_dims, use_embed=True):
    # - use_embed == 0: default positional encoding
    # - use_embed == -1: no positional encoding
    if not use_embed:
        return torch.nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


@misc.profiled_function
def compute_pos_enc(xyz_coords, pos_enc_fn, tex_h, tex_w, only_z=False):

    if only_z:
        n_coords = 1
        tex_h = 1
        tex_w = 1
    else:
        n_coords = 3

    misc.assert_shape(xyz_coords, [None, tex_h, tex_w, n_coords])

    n_planes = xyz_coords.shape[0]
    flat_xyz_coords = xyz_coords.reshape((-1, n_coords))

    flat_pos_encs = []
    for i in range(n_coords):
        # we need to process coordinates one axis by one axis
        flat_pos_encs.append(pos_enc_fn(flat_xyz_coords[:, i:(i + 1)]))
    # [#points, 3, pos_enc_dim]
    flat_pos_encs = torch.stack(flat_pos_encs, dim=1)
    assert flat_pos_encs.ndim == 3, f'{flat_pos_encs.shape}'

    # [#planes, H, W, 3, feat_dim] -> [#planes, 3, feat_dim, H, W]
    pos_encs = flat_pos_encs.reshape(
        (n_planes, tex_h, tex_w, n_coords, -1)).permute(0, 3, 4, 1, 2)

    return pos_encs.unsqueeze(0)


#----------------------------------------------------------------------------


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


#----------------------------------------------------------------------------


@misc.profiled_function
def modulated_conv2d(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,  # Optional noise tensor to add to the output activations.
    up=1,  # Integer upsampling factor.
    down=1,  # Integer downsampling factor.
    padding=0,  # Padding with respect to the upsampled image.
    resample_filter=None,  # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,  # Apply weight demodulation?
    flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(
            float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(
            float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x,
                        dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1),
                        noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(
    ):  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
        flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):

    def __init__(
            self,
            in_features,  # Number of input features.
            out_features,  # Number of output features.
            bias=True,  # Apply additive bias before the activation function?
            activation='linear',  # Activation function: 'relu', 'lrelu', etc.
            lr_multiplier=1,  # Learning rate multiplier.
            bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(
            torch.full([out_features],
                       np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):

    def __init__(
            self,
            in_channels,  # Number of input channels.
            out_channels,  # Number of output channels.
            kernel_size,  # Width and height of the convolution kernel.
            bias=True,  # Apply additive bias before the activation function?
            activation='linear',  # Activation function: 'relu', 'lrelu', etc.
            up=1,  # Integer upsampling factor.
            down=1,  # Integer downsampling factor.
            resample_filter=[
                1, 3, 3, 1
            ],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
            channels_last=False,  # Expect the input to have memory_format=channels_last?
            trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.channels_last = channels_last

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn(
            [out_channels, in_channels, kernel_size,
             kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(
            x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):

    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features
                         ] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(
                    dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                # [B, num_ws, w_dim]
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi)
        return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):

    def __init__(
            self,
            in_channels,  # Number of input channels.
            out_channels,  # Number of output channels.
            w_dim,  # Intermediate latent (W) dimensionality.
            resolution,  # Resolution of this layer.
            kernel_size=3,  # Convolution kernel size.
            up=1,  # Integer upsampling factor.
            use_noise=True,  # Enable noise input?
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            resample_filter=[
                1, 3, 3, 1
            ],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            channels_last=False,  # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size,
                         kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const',
                                 torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(
            x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution],
                device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp)
        return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim,
                 kernel_size=1,
                 conv_clamp=None,
                 channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size,
                         kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w, fused_modconv=True, splitted=False, n_planes=1):
        bs = w.shape[0]
        styles = self.affine(w) * self.weight_gain
        if splitted:
            # We make generation plane-specific. Therefore, we need to duplicate styles tensor.
            styles = styles.unsqueeze(1).expand(-1, n_planes, -1).reshape(
                (bs * n_planes, -1))
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            demodulate=False,
            fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


@persistence.persistent_class
class ToRGBLayerDeeper(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim,
                 kernel_size=1,
                 intermediate_channels=16,
                 conv_clamp=None,
                 channels_last=False,
                 act_name='lrelu',
                 resample_filter=[1, 3, 3, 1]):
        super().__init__()
        self.conv_clamp = conv_clamp
        memory_format = torch.channels_last if channels_last else torch.contiguous_format

        self.affine1 = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight1 = torch.nn.Parameter(
            torch.randn(
                [intermediate_channels, in_channels, kernel_size,
                 kernel_size]).to(memory_format=memory_format))
        self.bias1 = torch.nn.Parameter(torch.zeros([intermediate_channels]))
        self.weight_gain1 = 1 / np.sqrt(in_channels * (kernel_size**2))

        self.conv = torch.nn.Sequential(
            Conv2dLayer(
                intermediate_channels,
                intermediate_channels,
                kernel_size=kernel_size,
                bias=True,
                up=1,
                activation=act_name,
                resample_filter=resample_filter,
                channels_last=channels_last),
            Conv2dLayer(
                intermediate_channels,
                intermediate_channels,
                kernel_size=kernel_size,
                bias=True,
                up=1,
                activation=act_name,
                resample_filter=resample_filter,
                channels_last=channels_last),
            Conv2dLayer(
                intermediate_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=True,
                up=1,
                activation=act_name,
                resample_filter=resample_filter,
                channels_last=channels_last),
        )

    def forward(self, x, w, fused_modconv=True, splitted=False, n_planes=1):
        bs = w.shape[0]
        styles1 = self.affine1(w) * self.weight_gain1
        if splitted:
            styles1 = styles1.unsqueeze(1).expand(-1, n_planes, -1).reshape(
                (bs * n_planes, -1))
        x = modulated_conv2d(
            x=x,
            weight=self.weight1,
            styles=styles1,
            demodulate=False,
            fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias1.to(x.dtype), clamp=None)

        x = self.conv(x)

        return x


@persistence.persistent_class
class ToRGBLayerDeeperModulatedConv(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim,
                 kernel_size=1,
                 intermediate_channels=16,
                 conv_clamp=None,
                 channels_last=False,
                 act_name='lrelu'):
        super().__init__()
        self.conv_clamp = conv_clamp
        memory_format = torch.channels_last if channels_last else torch.contiguous_format

        self.act_name = act_name

        if isinstance(intermediate_channels, int):
            intermediate_ch_list = [intermediate_channels for _ in range(3)]
        elif isinstance(intermediate_channels, list):
            intermediate_ch_list = intermediate_channels
        else:
            raise ValueError(f'{type(intermediate_channels)}')

        print('\ntoRGB: ', in_channels, intermediate_ch_list, out_channels,
              '\n')

        self.affine1 = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight1 = torch.nn.Parameter(
            torch.randn([
                intermediate_ch_list[0], in_channels, kernel_size, kernel_size
            ]).to(memory_format=memory_format))
        self.bias1 = torch.nn.Parameter(torch.zeros([intermediate_ch_list[0]]))
        self.weight_gain1 = 1 / np.sqrt(in_channels * (kernel_size**2))

        self.affine2 = FullyConnectedLayer(
            w_dim, intermediate_ch_list[0], bias_init=1)
        self.weight2 = torch.nn.Parameter(
            torch.randn([
                intermediate_ch_list[1], intermediate_ch_list[0], kernel_size,
                kernel_size
            ]).to(memory_format=memory_format))
        self.bias2 = torch.nn.Parameter(torch.zeros([intermediate_ch_list[1]]))
        self.weight_gain2 = 1 / np.sqrt(intermediate_ch_list[0] *
                                        (kernel_size**2))

        self.affine3 = FullyConnectedLayer(
            w_dim, intermediate_ch_list[1], bias_init=1)
        self.weight3 = torch.nn.Parameter(
            torch.randn([
                intermediate_ch_list[2], intermediate_ch_list[1], kernel_size,
                kernel_size
            ]).to(memory_format=memory_format))
        self.bias3 = torch.nn.Parameter(torch.zeros([intermediate_ch_list[2]]))
        self.weight_gain3 = 1 / np.sqrt(intermediate_ch_list[1] *
                                        (kernel_size**2))

        self.affine4 = FullyConnectedLayer(
            w_dim, intermediate_ch_list[2], bias_init=1)
        self.weight4 = torch.nn.Parameter(
            torch.randn([
                out_channels, intermediate_ch_list[2], kernel_size, kernel_size
            ]).to(memory_format=memory_format))
        self.bias4 = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain4 = 1 / np.sqrt(intermediate_ch_list[2] *
                                        (kernel_size**2))

    def forward(self, x, w, fused_modconv=True, splitted=False, n_planes=1):
        bs = w.shape[0]
        styles1 = self.affine1(w) * self.weight_gain1
        if splitted:
            styles1 = styles1.unsqueeze(1).expand(-1, n_planes,
                                                  -1).contiguous().reshape(
                                                      (bs * n_planes, -1))
        x = modulated_conv2d(
            x=x,
            weight=self.weight1,
            styles=styles1,
            demodulate=False,
            fused_modconv=fused_modconv)
        x = bias_act.bias_act(
            x,
            self.bias1.to(x.dtype),
            clamp=self.conv_clamp,
            act=self.act_name)

        styles2 = self.affine2(w) * self.weight_gain2
        if splitted:
            styles2 = styles2.unsqueeze(1).expand(-1, n_planes,
                                                  -1).contiguous().reshape(
                                                      (bs * n_planes, -1))
        x = modulated_conv2d(
            x=x,
            weight=self.weight2,
            styles=styles2,
            demodulate=False,
            fused_modconv=fused_modconv)
        x = bias_act.bias_act(
            x,
            self.bias2.to(x.dtype),
            clamp=self.conv_clamp,
            act=self.act_name)

        styles3 = self.affine3(w) * self.weight_gain3
        if splitted:
            styles3 = styles3.unsqueeze(1).expand(-1, n_planes,
                                                  -1).contiguous().reshape(
                                                      (bs * n_planes, -1))
        x = modulated_conv2d(
            x=x,
            weight=self.weight3,
            styles=styles3,
            demodulate=False,
            fused_modconv=fused_modconv)
        x = bias_act.bias_act(
            x,
            self.bias3.to(x.dtype),
            clamp=self.conv_clamp,
            act=self.act_name)

        styles4 = self.affine4(w) * self.weight_gain4
        if splitted:
            styles4 = styles4.unsqueeze(1).expand(-1, n_planes,
                                                  -1).contiguous().reshape(
                                                      (bs * n_planes, -1))
        x = modulated_conv2d(
            x=x,
            weight=self.weight4,
            styles=styles4,
            demodulate=False,
            fused_modconv=fused_modconv)
        x = bias_act.bias_act(
            x,
            self.bias4.to(x.dtype),
            clamp=self.conv_clamp,
            act=self.act_name)

        return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):

    def __init__(
            self,
            in_channels,  # Number of input channels, 0 = first block.
            out_channels,  # Number of output channels.
            w_dim,  # Intermediate latent (W) dimensionality.
            resolution,  # Resolution of this block.
            img_channels,  # Number of output color channels.
            is_last,  # Is this the last block?
            architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
            resample_filter=[
                1, 3, 3, 1
            ],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            use_fp16=False,  # Use FP16 for this block?
            fp16_channels_last=False,  # Use channels-last memory format with FP16?
            # MPI
        pos_enc_multires=0,  # Number of channels for positional encoding.
            torgba_cond_on_pos_enc='normalize_add_z',  # Whether to condition on Z or XYZ.
            torgba_cond_on_pos_enc_embed_func='modulated_lrelu',
            # how to produce MPI's RGB-a
            torgba_sep_background=False,  # Whether to generate background and foreground separately.
            build_background_from_rgb=False,  # Whether to build background image from boundaries of RGB.
            build_background_from_rgb_ratio=0.05,
            cond_on_pos_enc_only_alpha=False,  # Whether to only use "cond_on_pos_enc" for alpha channels.
            gen_alpha_largest_res=256,  # Largest resolution to generate alpha maps.
            **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.torgba_cond_on_pos_enc = torgba_cond_on_pos_enc
        self.torgba_cond_on_pos_enc_embed_func = torgba_cond_on_pos_enc_embed_func
        self.torgba_sep_background = torgba_sep_background
        self.build_background_from_rgb = build_background_from_rgb
        self.build_background_from_rgb_ratio = build_background_from_rgb_ratio
        self.cond_on_pos_enc_only_alpha = cond_on_pos_enc_only_alpha
        self.gen_alpha_largest_res = gen_alpha_largest_res

        self.gen_alpha_this_res = self.gen_alpha_largest_res >= self.resolution

        # NOTE: this network will only process cond_on_pos_enc
        assert self.torgba_cond_on_pos_enc not in ['none']

        if self.build_background_from_rgb:
            assert self.torgba_sep_background

        assert self.torgba_cond_on_pos_enc in [
            'none', 'cond_z', 'cond_xyz', 'cat_xyz', 'add_z',
            'normalize_add_z', 'add_xyz', 'normalize_add_xyz'
        ], f'{self.torgba_cond_on_pos_enc}'

        self.pos_enc_fn, pos_enc_ch_single_embed = get_embedder(
            pos_enc_multires, 1, use_embed=True)

        if self.torgba_cond_on_pos_enc in [
                'cond_z', 'add_z', 'normalize_add_z', 'add_xyz',
                'normalize_add_xyz'
        ]:
            self.pos_enc_total_ch = pos_enc_ch_single_embed
        elif self.torgba_cond_on_pos_enc in ['cond_xyz', 'cat_xyz']:
            self.pos_enc_total_ch = pos_enc_ch_single_embed * 3
        else:
            raise ValueError

        print('\nconv_clamp: ', conv_clamp, '\n')

        if self.gen_alpha_this_res:
            if self.torgba_cond_on_pos_enc in ['add_xyz', 'normalize_add_xyz']:
                if 'mlp' in self.torgba_cond_on_pos_enc_embed_func:
                    self.pos_enc_embed_x = self._mlp(
                        self.pos_enc_total_ch,
                        out_channels,
                        activation='linear')
                    self.pos_enc_embed_y = self._mlp(
                        self.pos_enc_total_ch,
                        out_channels,
                        activation='linear')
                    self.pos_enc_embed_z = self._mlp(
                        self.pos_enc_total_ch,
                        out_channels,
                        activation='linear')
                elif 'conv' in self.torgba_cond_on_pos_enc_embed_func:
                    act_name = self.torgba_cond_on_pos_enc_embed_func.split(
                        '_')[1]
                    print('\nact_name: ', act_name, '\n')
                    self.pos_enc_embed_x = self._conv(
                        self.pos_enc_total_ch,
                        out_channels,
                        conv_clamp=conv_clamp,
                        act_name=act_name,
                        resample_filter=resample_filter
                    )  # deeper=pos_enc_layer_deeper)
                    self.pos_enc_embed_y = self._conv(
                        self.pos_enc_total_ch,
                        out_channels,
                        conv_clamp=conv_clamp,
                        act_name=act_name,
                        resample_filter=resample_filter
                    )  # deeper=pos_enc_layer_deeper)
                    self.pos_enc_embed_z = self._conv(
                        self.pos_enc_total_ch,
                        out_channels,
                        conv_clamp=conv_clamp,
                        act_name=act_name,
                        resample_filter=resample_filter
                    )  # deeper=pos_enc_layer_deeper)
                elif 'modulated' in self.torgba_cond_on_pos_enc_embed_func:
                    act_name = self.torgba_cond_on_pos_enc_embed_func.split(
                        '_')[1]
                    print('\nact_name: ', act_name, '\n')
                    intermediate_ch_list = [
                        out_channels // 4, out_channels // 2, out_channels
                    ]
                    self.pos_enc_embed_x = ToRGBLayerDeeperModulatedConv(
                        self.pos_enc_total_ch,
                        out_channels,
                        intermediate_channels=intermediate_ch_list,
                        w_dim=w_dim,
                        act_name=act_name,
                        conv_clamp=conv_clamp,
                        channels_last=self.channels_last)
                    self.pos_enc_embed_y = ToRGBLayerDeeperModulatedConv(
                        self.pos_enc_total_ch,
                        out_channels,
                        intermediate_channels=intermediate_ch_list,
                        w_dim=w_dim,
                        act_name=act_name,
                        conv_clamp=conv_clamp,
                        channels_last=self.channels_last)
                    self.pos_enc_embed_z = ToRGBLayerDeeperModulatedConv(
                        self.pos_enc_total_ch,
                        out_channels,
                        intermediate_channels=intermediate_ch_list,
                        w_dim=w_dim,
                        act_name=act_name,
                        conv_clamp=conv_clamp,
                        channels_last=self.channels_last)
                else:
                    raise ValueError
            else:
                if 'mlp' in self.torgba_cond_on_pos_enc_embed_func:
                    self.pos_enc_embed = self._mlp(
                        self.pos_enc_total_ch,
                        out_channels,
                        activation='linear')
                elif 'conv' in self.torgba_cond_on_pos_enc_embed_func:
                    act_name = self.torgba_cond_on_pos_enc_embed_func.split(
                        '_')[1]
                    print('\nact_name: ', act_name, '\n')
                    self.pos_enc_embed = self._conv(
                        self.pos_enc_total_ch,
                        out_channels,
                        conv_clamp=conv_clamp,
                        act_name=act_name,
                        resample_filter=resample_filter
                    )  # deeper=pos_enc_layer_deeper)
                elif 'modulated' in self.torgba_cond_on_pos_enc_embed_func:
                    act_name = self.torgba_cond_on_pos_enc_embed_func.split(
                        '_')[1]
                    print('\nact_name: ', act_name, '\n')
                    intermediate_ch_list = [
                        out_channels // 4, out_channels // 2, out_channels
                    ]
                    self.pos_enc_embed = ToRGBLayerDeeperModulatedConv(
                        self.pos_enc_total_ch,
                        out_channels,
                        intermediate_channels=intermediate_ch_list,
                        w_dim=w_dim,
                        act_name=act_name,
                        conv_clamp=conv_clamp,
                        channels_last=self.channels_last)
                else:
                    raise ValueError

        if in_channels == 0:
            self.const = torch.nn.Parameter(
                torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=2,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            assert self.img_channels == 4, f'{self.img_channels}'

            if self.torgba_cond_on_pos_enc == 'cat_xyz':
                extra_channels = self.pos_enc_total_ch
            else:
                extra_channels = 0

            if self.torgba_sep_background:
                pass
            else:
                self.tobackground = None

            if self.cond_on_pos_enc_only_alpha:

                self.torgb = ToRGBLayer(
                    out_channels,
                    3,
                    w_dim=w_dim,
                    conv_clamp=conv_clamp,
                    channels_last=self.channels_last)

                if self.gen_alpha_this_res:

                    self.toalpha = ToRGBLayer(
                        out_channels + extra_channels,
                        1,
                        w_dim=w_dim,
                        conv_clamp=conv_clamp,
                        channels_last=self.channels_last)
                else:
                    self.toalpha = None
            else:
                self.torgba = ToRGBLayer(
                    out_channels + extra_channels,
                    self.img_channels,
                    w_dim=w_dim,
                    conv_clamp=conv_clamp,
                    channels_last=self.channels_last)

            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last)

    def _mlp(self, pos_enc_total_ch, out_channels, activation='linear'):
        return FullyConnectedLayer(
            pos_enc_total_ch, out_channels, activation=activation)

    def _conv(self,
              pos_enc_total_ch,
              out_channels,
              conv_clamp=None,
              act_name='linear',
              resample_filter=[1, 3, 3, 1],
              deeper=False):
        assert out_channels // 4 >= pos_enc_total_ch, f'{pos_enc_total_ch}, {out_channels // 4}'
        if deeper:
            pos_enc_embed = torch.nn.Sequential(
                # 1x1
                Conv2dLayer(
                    pos_enc_total_ch,
                    out_channels // 4,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
                # 1x1
                Conv2dLayer(
                    out_channels // 4,
                    out_channels // 2,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
                # 1x1
                Conv2dLayer(
                    out_channels // 2,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
                # 1x1
                Conv2dLayer(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
                # 1x1
                Conv2dLayer(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
                # 1x1
                Conv2dLayer(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
            )
        else:
            pos_enc_embed = torch.nn.Sequential(
                # 1x1
                Conv2dLayer(
                    pos_enc_total_ch,
                    out_channels // 4,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
                # 1x1
                Conv2dLayer(
                    out_channels // 4,
                    out_channels // 2,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
                # 1x1
                Conv2dLayer(
                    out_channels // 2,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    up=1,
                    conv_clamp=conv_clamp,
                    activation=act_name,
                    resample_filter=resample_filter,
                    channels_last=self.channels_last),
            )
        return pos_enc_embed

    def forward(self,
                x,
                img,
                ws,
                force_fp32=False,
                fused_modconv=None,
                xyz_coords=None,
                xyz_coords_only_z=False,
                enable_feat_net_grad=True,
                n_planes=32,
                **layer_kwargs):

        self.n_planes = n_planes

        bs = ws.shape[0]

        with torch.set_grad_enabled(enable_feat_net_grad):

            misc.assert_shape(
                ws, [None, self.num_conv + self.num_torgb, self.w_dim])
            w_iter = iter(ws.unbind(dim=1))
            dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
            memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
            if fused_modconv is None:
                with misc.suppress_tracer_warnings(
                ):  # this value will be treated as a constant
                    fused_modconv = (not self.training) and (
                        dtype == torch.float32 or int(x.shape[0]) == 1)

            # Input.
            if self.in_channels == 0:
                x = self.const.to(dtype=dtype, memory_format=memory_format)
                x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
            else:
                misc.assert_shape(x, [
                    None, self.in_channels, self.resolution // 2,
                    self.resolution // 2
                ])
                x = x.to(dtype=dtype, memory_format=memory_format)

            # Main layers.
            if self.in_channels == 0:
                w_conv1 = next(w_iter)
                x = self.conv1(
                    x, w_conv1, fused_modconv=fused_modconv, **layer_kwargs)
            elif self.architecture == 'resnet':
                y = self.skip(x, gain=np.sqrt(0.5))
                w_conv0 = next(w_iter)
                x = self.conv0(
                    x, w_conv0, fused_modconv=fused_modconv, **layer_kwargs)
                w_conv1 = next(w_iter)
                x = self.conv1(
                    x,
                    w_conv1,
                    fused_modconv=fused_modconv,
                    gain=np.sqrt(0.5),
                    **layer_kwargs)
                x = y.add_(x)
            else:
                w_conv0 = next(w_iter)
                x = self.conv0(
                    x, w_conv0, fused_modconv=fused_modconv, **layer_kwargs)
                w_conv1 = next(w_iter)
                x = self.conv1(
                    x, w_conv1, fused_modconv=fused_modconv, **layer_kwargs)

        # print("\nx: ", x.shape, x.dtype)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [
                None, self.img_channels * self.n_planes, self.resolution // 2,
                self.resolution // 2
            ])

            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.is_last or self.architecture == 'skip':

            if self.gen_alpha_this_res and self.torgba_cond_on_pos_enc != 'none':
                assert xyz_coords is not None
                # [1, #planes, 3, pos_enc_ch, H, W]
                xyz_pos_encs = compute_pos_enc(
                    xyz_coords,
                    self.pos_enc_fn,
                    self.resolution,
                    self.resolution,
                    only_z=xyz_coords_only_z)
                xyz_pos_encs = xyz_pos_encs.to(
                    dtype=dtype, memory_format=memory_format)

                if self.torgba_cond_on_pos_enc in [
                        'cond_z', 'add_z', 'normalize_add_z'
                ]:
                    # [1, #planes, pos_enc_ch, H, W]
                    if xyz_coords_only_z:
                        xyz_pos_encs = xyz_pos_encs[:, :, 0, ...]
                    else:
                        xyz_pos_encs = xyz_pos_encs[:, :, 2, ...]
                elif self.torgba_cond_on_pos_enc in ['cond_xyz', 'cat_xyz']:
                    # [1, #planes, pos_enc_ch, H, W]
                    xyz_pos_encs = xyz_pos_encs.reshape(
                        (1, self.n_planes, -1, self.resolution,
                         self.resolution))
                elif self.torgba_cond_on_pos_enc in [
                        'add_xyz', 'normalize_add_xyz'
                ]:
                    assert not xyz_coords_only_z
                    # [1, #planes, pos_enc_ch, H, W]
                    x_pos_encs = xyz_pos_encs[:, :, 0, ...]
                    y_pos_encs = xyz_pos_encs[:, :, 1, ...]
                    z_pos_encs = xyz_pos_encs[:, :, 2, ...]
                else:
                    raise ValueError

                if self.torgba_cond_on_pos_enc in ['cond_z', 'cond_xyz']:
                    # # [bs, 1, feat_dim, H, W] -> [bs, #planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
                    # cond_x = x.unsqueeze(1).expand(-1, self.n_planes, -1, -1, -1).reshape((bs * self.n_planes, -1, self.resolution, self.resolution))

                    if 'mlp' in self.torgba_cond_on_pos_enc_embed_func:
                        # [1, #planes, pos_enc_ch, H, W] -> [1, #planes, H, W, pos_enc_ch] -> [#planes x H x W, pos_enc_ch]
                        flat_xyz_pos_encs = xyz_pos_encs.permute(
                            0, 1, 3, 4, 2).reshape(
                                (self.n_planes * self.resolution**2,
                                 self.pos_enc_total_ch))
                        # [#planes x H x W, feat_dim]
                        flat_xyz_pos_enc_embeds = self.pos_enc_embed(
                            flat_xyz_pos_encs)
                        # [#planes, H, W, feat_dim] -> [#planes, feat_dim, H, W]
                        xyz_pos_enc_embeds = flat_xyz_pos_enc_embeds.view(
                            (self.n_planes, self.resolution, self.resolution,
                             -1)).permute(0, 3, 1, 2)

                        # # [bs x #planes, feat_dim, H, W]
                        # xyz_pos_enc_embeds = xyz_pos_enc_embeds.repeat(bs, 1, 1, 1)
                    elif 'conv' in self.torgba_cond_on_pos_enc_embed_func:
                        # [#planes, feat_dim, H, W]
                        xyz_pos_enc_embeds = self.pos_enc_embed(
                            xyz_pos_encs[0, ...])

                        # # [bs x #planes, feat_dim, H, W]
                        # xyz_pos_enc_embeds = xyz_pos_enc_embeds.repeat(bs, 1, 1, 1)
                    else:
                        raise ValueError

                    cond_x = x

                    # [bs, feat_dim, 1, 1]
                    cond_x_mean, cond_x_std = calc_mean_std(cond_x)
                    cond_x = (cond_x - cond_x_mean) / cond_x_std

                    # [bs, 1, feat_dim, H, W] -> [bs, #planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
                    cond_x = cond_x.unsqueeze(1).expand(
                        -1, self.n_planes, -1, -1, -1).reshape(
                            (bs * self.n_planes, -1, self.resolution,
                             self.resolution))

                    # [#planes, feat_dim, 1, 1]
                    xyz_pos_enc_embeds_mean, xyz_pos_enc_embeds_std = calc_mean_std(
                        xyz_pos_enc_embeds)

                    # [bs x #planes, feat_dim, H, W]
                    xyz_pos_enc_embeds_mean = xyz_pos_enc_embeds_mean.repeat(
                        bs, 1, 1, 1)
                    xyz_pos_enc_embeds_std = xyz_pos_enc_embeds_std.repeat(
                        bs, 1, 1, 1)

                    cond_x = cond_x * xyz_pos_enc_embeds_std + xyz_pos_enc_embeds_mean

                    # cond_x = adaptive_instance_normalization(cond_x, xyz_pos_enc_embeds)
                elif self.torgba_cond_on_pos_enc == 'cat_xyz':
                    # [bs, 1, feat_dim, H, W] -> [bs, #planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
                    cond_x = x.unsqueeze(1).expand(-1, self.n_planes, -1, -1,
                                                   -1).reshape(
                                                       (bs * self.n_planes, -1,
                                                        self.resolution,
                                                        self.resolution))

                    # [bs x #planes, pos_enc_dim, H, W]
                    rep_xyz_pos_encs = xyz_pos_encs[0, ...].repeat(bs, 1, 1, 1)
                    cond_x = torch.cat((cond_x, rep_xyz_pos_encs), dim=1)
                elif self.torgba_cond_on_pos_enc in [
                        'add_z', 'normalize_add_z'
                ]:
                    xyz_pos_enc_embeds = self._add_z(
                        bs,
                        xyz_pos_encs,
                        self.pos_enc_embed,
                        w=w_conv1,
                        fused_modconv=fused_modconv)

                    cond_x = x

                    if self.torgba_cond_on_pos_enc == 'normalize_add_z':
                        # [bs, feat_dim, 1, 1]
                        cond_x_mean, cond_x_std = calc_mean_std(cond_x)
                        cond_x = (cond_x - cond_x_mean) / (
                            cond_x_std + FLOATING_EPS)

                    # [bs, 1, feat_dim, H, W] -> [bs, #planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
                    cond_x = cond_x.unsqueeze(1).expand(
                        -1, self.n_planes, -1, -1, -1).reshape(
                            (bs * self.n_planes, -1, self.resolution,
                             self.resolution))

                    cond_x = cond_x + xyz_pos_enc_embeds
                elif self.torgba_cond_on_pos_enc in [
                        'add_xyz', 'normalize_add_xyz'
                ]:
                    x_pos_enc_embeds = self._add_x(
                        bs,
                        x_pos_encs,
                        self.pos_enc_embed_x,
                        w=w_conv1,
                        fused_modconv=fused_modconv)
                    y_pos_enc_embeds = self._add_y(
                        bs,
                        y_pos_encs,
                        self.pos_enc_embed_y,
                        w=w_conv1,
                        fused_modconv=fused_modconv)
                    z_pos_enc_embeds = self._add_z(
                        bs,
                        z_pos_encs,
                        self.pos_enc_embed_z,
                        w=w_conv1,
                        fused_modconv=fused_modconv)

                    cond_x = x

                    if self.torgba_cond_on_pos_enc == 'normalize_add_xyz':
                        # [bs, feat_dim, 1, 1]
                        cond_x_mean, cond_x_std = calc_mean_std(cond_x)
                        cond_x = (cond_x - cond_x_mean) / (
                            cond_x_std + FLOATING_EPS)

                    # [bs, 1, feat_dim, H, W] -> [bs, #planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
                    cond_x = cond_x.unsqueeze(1).expand(
                        -1, self.n_planes, -1, -1, -1).reshape(
                            (bs * self.n_planes, -1, self.resolution,
                             self.resolution))

                    cond_x = cond_x + x_pos_enc_embeds + y_pos_enc_embeds + z_pos_enc_embeds
                else:
                    raise ValueError
            else:
                cond_x = None

            # NOTE: we use same w for all toRGB layers
            w_for_rgba = next(w_iter)

            w_background = w_for_rgba
            w_rgb = w_for_rgba
            w_alpha = w_for_rgba
            w_rgba = w_for_rgba

            if self.torgba_sep_background:
                if self.build_background_from_rgb:
                    with torch.no_grad():
                        if self.resolution >= 0:
                            # NOTE: we use elements from left/top/right boundaries.
                            # We do not use elements near bottom boundary as there are necks.
                            cur_background_x = torch.zeros(
                                x.shape,
                                device=x.device,
                                dtype=dtype,
                                requires_grad=False)
                            tmp_pad = max(
                                1,
                                int(
                                    np.floor(
                                        self.build_background_from_rgb_ratio *
                                        self.resolution)))
                            # left
                            cur_background_x[:, :, :, :
                                             tmp_pad] = x[:, :, :, :
                                                          tmp_pad].detach()
                            # right
                            cur_background_x[:, :, :,
                                             -tmp_pad:] = x[:, :, :,
                                                            -tmp_pad:].detach(
                                                            )
                            # # top
                            # cur_background_x[:, :, :tmp_pad, :] = x[:, :, :tmp_pad, :].detach()
                            tmp_top = 0
                            # interpolation
                            tmp_start_col = tmp_pad
                            tmp_end_col = self.resolution - tmp_pad
                            if tmp_start_col < tmp_end_col:
                                tmp_cols = torch.arange(
                                    tmp_start_col,
                                    tmp_end_col,
                                    device=x.device).reshape((1, 1, 1, -1))
                                tmp_right_ratios = (
                                    tmp_cols - tmp_start_col) / (
                                        tmp_end_col - tmp_start_col +
                                        FLOATING_EPS)
                                # [bs, feat_dim, #rows, #columns]
                                tmp_left_feat = x[:, :, tmp_top:,
                                                  tmp_start_col:(
                                                      tmp_start_col +
                                                      1)].detach()
                                tmp_right_feat = x[:, :, tmp_top:,
                                                   (tmp_end_col -
                                                    1):tmp_end_col].detach()
                                tmp_interpolated = (
                                    1 - tmp_right_ratios
                                ) * tmp_left_feat + tmp_right_ratios * tmp_right_feat
                                cur_background_x[:, :, tmp_top:, tmp_start_col:
                                                 tmp_end_col] = tmp_interpolated
                        else:
                            cur_background_x = x.detach()
                    cur_background = self.torgb(
                        cur_background_x,
                        w_rgb,
                        fused_modconv=fused_modconv,
                        splitted=False,
                        n_planes=self.n_planes)
                else:
                    raise NotImplementedError

                cur_background = cur_background.unsqueeze(1)
            else:
                cur_background = None
            if self.cond_on_pos_enc_only_alpha:
                # [bs, 3, H, W]
                # w_rgb = next(w_iter)
                single_rgb = self.torgb(
                    x,
                    w_rgb,
                    fused_modconv=fused_modconv,
                    splitted=False,
                    n_planes=self.n_planes)

                # print("single_rgb: ", single_rgb.shape, single_rgb.dtype, "\n\n")

                if self.torgba_sep_background:
                    # [bs, 3, H, W] -> [bs, 1, 3, H, W] -> [bs, #planes - 1, 3, H, W]
                    cur_rgb = single_rgb.unsqueeze(1).expand(
                        -1, self.n_planes - 1, -1, -1, -1)
                    # [bs, #planes, 3, H, W]
                    cur_rgb = torch.cat((cur_rgb, cur_background), dim=1)
                else:
                    # [bs, 3, H, W] -> [bs, 1, 3, H, W] -> [bs, #planes - 1, 3, H, W]
                    cur_rgb = single_rgb.unsqueeze(1).expand(
                        -1, self.n_planes, -1, -1, -1)

                # [bs, #planes, 3, H, W] -> [bs x #planes, 3, H, W]
                cur_rgb = cur_rgb.reshape(
                    (bs * self.n_planes, 3, self.resolution, self.resolution))

                if self.gen_alpha_this_res:
                    # [bs x #planes, 1, H, W]
                    # w_alpha = next(w_iter)
                    cur_alpha = self.toalpha(
                        cond_x,
                        w_alpha,
                        fused_modconv=fused_modconv,
                        splitted=True,
                        n_planes=self.n_planes)
                else:
                    # NOTE: we do not generate alpha maps at this resolution
                    cur_alpha = torch.zeros((bs * self.n_planes, 1,
                                             self.resolution, self.resolution),
                                            device=cur_rgb.device)

                # [bs x #planes, 4, H, W]
                y = torch.cat((cur_rgb, cur_alpha), dim=1)
            else:
                if self.torgba_sep_background:
                    raise ValueError(
                        '\nWe do not support combined separate background and depth-aware to_RGBa layer.\n'
                    )
                else:
                    # [bs x #planes, 4, H, W]
                    # w_rgba = next(w_iter)
                    y = self.torgba(
                        cond_x,
                        w_rgba,
                        fused_modconv=fused_modconv,
                        splitted=True,
                        n_planes=self.n_planes)

            y = y.reshape(
                (bs, self.n_planes, 4, self.resolution, self.resolution)).view(
                    (bs, self.n_planes * 4, self.resolution, self.resolution))
            # [B, #planes, 4, H, W] -> [B, #planes * 4, H, W]
            y = y.to(
                dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def _add_z(self,
               bs,
               xyz_pos_encs,
               pos_enc_embed,
               w=None,
               fused_modconv=False):
        # [#planes, pos_enc_ch, 1, 1], since Z is same for whole plane, we only need one elements
        selected_xyz_pos_encs = xyz_pos_encs[0, :, :, :1, :1]
        if 'mlp' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes, pos_enc_ch, 1, 1] -> [#planes, pos_enc_ch]
            flat_xyz_pos_encs = selected_xyz_pos_encs.reshape(
                (self.n_planes, self.pos_enc_total_ch))
            # [#planes, feat_dim]
            flat_xyz_pos_enc_embeds = pos_enc_embed(flat_xyz_pos_encs)
            # [#planes, feat_dim] -> [#planes, feat_dim, 1, 1] -> [#planes, feat_dim, H, W]
            xyz_pos_enc_embeds = flat_xyz_pos_enc_embeds.unsqueeze(
                -1).unsqueeze(-1)
            # [#planes, feat_dim, 1, 1] -> [#planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(
                -1, -1, self.resolution, self.resolution)
            # [#planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.repeat(bs, 1, 1, 1)
        elif 'conv' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes, feat_dim, 1, 1]
            xyz_pos_enc_embeds = pos_enc_embed(selected_xyz_pos_encs)

            # # NOTE: DEBUG
            # import os
            # os.makedirs("/mnt/playground/ckpts/20220207/gy5vrijptd/xyz_pos_enc_embeds", exist_ok=True)
            # np.save(f"/mnt/playground/ckpts/20220207/gy5vrijptd/xyz_pos_enc_embeds/xyz_pos_enc_embeds_{self.resolution:03d}.npy", xyz_pos_enc_embeds.detach().cpu().numpy())

            # [#planes, feat_dim, 1, 1] -> [#planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(
                -1, -1, self.resolution, self.resolution)
            # [bs x #planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.repeat(bs, 1, 1, 1)
        elif 'modulated' in self.torgba_cond_on_pos_enc_embed_func:
            # [bs, #planes, pos_enc_ch, 1, 1] -> [bs x #planes, pos_enc_ch, 1, 1]
            selected_xyz_pos_encs = selected_xyz_pos_encs.unsqueeze(0).expand(
                bs, -1, -1, -1, -1)
            selected_xyz_pos_encs = selected_xyz_pos_encs.reshape(
                (bs * self.n_planes, self.pos_enc_total_ch, 1, 1))
            xyz_pos_enc_embeds = pos_enc_embed(
                selected_xyz_pos_encs,
                w,
                fused_modconv=fused_modconv,
                splitted=True,
                n_planes=self.n_planes)

            # [bs x #planes, feat_dim, 1, 1] -> [bs x #planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(
                -1, -1, self.resolution, self.resolution)
        else:
            raise ValueError
        return xyz_pos_enc_embeds

    def _add_x(self,
               bs,
               xyz_pos_encs,
               pos_enc_embed,
               w=None,
               fused_modconv=False):
        # [#planes, pos_enc_ch, 1, W], each plane has same X coords
        selected_xyz_pos_encs = xyz_pos_encs[0, :, :, :1, :]
        # [#planes, pos_enc_ch, 1, W] -> [#planes, W, pos_enc_ch, 1] -> [#planes x W, pos_enc_ch]
        flat_xyz_pos_encs = selected_xyz_pos_encs.permute(0, 3, 1, 2).reshape(
            (self.n_planes * self.resolution, self.pos_enc_total_ch))
        if 'mlp' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes x W, feat_dim]
            flat_xyz_pos_enc_embeds = pos_enc_embed(flat_xyz_pos_encs)
        elif 'conv' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes x W, pos_enc_ch] -> [#planes x W, pos_enc_ch, 1, 1]
            flat_xyz_pos_encs = flat_xyz_pos_encs.unsqueeze(-1).unsqueeze(-1)
            # [#planes x W, feat_dim, 1, 1] -> [#planes x W, feat_dim]
            flat_xyz_pos_enc_embeds = pos_enc_embed(flat_xyz_pos_encs)[..., 0,
                                                                       0]
        elif 'modulated' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes, pos_enc_ch, 1, W] -> [W, #planes, pos_enc_ch, 1] -> [W x #planes, pos_enc_ch]
            flat_xyz_pos_encs = selected_xyz_pos_encs.permute(
                3, 0, 1, 2).reshape(
                    (self.resolution * self.n_planes, self.pos_enc_total_ch))
            # [bs, W x #planes, pos_enc_ch] -> [bs x W x #planes, pos_enc_ch]
            flat_xyz_pos_encs = flat_xyz_pos_encs.unsqueeze(0).expand(
                bs, -1, -1)
            flat_xyz_pos_encs = flat_xyz_pos_encs.reshape(
                (bs * self.resolution * self.n_planes, self.pos_enc_total_ch))
            # [bs x W x #planes, pos_enc_ch] -> [bs x W x #planes, pos_enc_ch, 1, 1]
            flat_xyz_pos_encs = flat_xyz_pos_encs.unsqueeze(-1).unsqueeze(-1)
            # [bs, dim] -> [bs, 1, dim] -> [bs, W, dim] -> [bs x W, dim]
            w = w.unsqueeze(1).expand(-1, self.resolution, -1).reshape(
                (bs * self.resolution, -1))
            # [bs x W x #planes, feat_dim, 1, 1] -> [bs x W x #planes, feat_dim]
            flat_xyz_pos_enc_embeds = pos_enc_embed(
                flat_xyz_pos_encs,
                w,
                fused_modconv=fused_modconv,
                splitted=True,
                n_planes=self.n_planes)[..., 0, 0]
        else:
            raise ValueError

        if 'modulated' not in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes x W, feat_dim] -> [#planes, W, feat_dim]
            xyz_pos_enc_embeds = flat_xyz_pos_enc_embeds.reshape(
                (self.n_planes, self.resolution, -1))
            # [#planes, W, feat_dim] -> [#planes, feat_dim, W] -> [#planes, feat_dim, 1, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.permute(0, 2,
                                                            1).unsqueeze(2)
            # [#planes, feat_dim, 1, W] -> [#planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(
                -1, -1, self.resolution, -1)
            # [#planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.repeat(bs, 1, 1, 1)
        else:
            # [bs x W x #planes, feat_dim] -> [bs, W, #planes, feat_dim]
            xyz_pos_enc_embeds = flat_xyz_pos_enc_embeds.reshape(
                (bs, self.resolution, self.n_planes, -1))
            # [bs, W, #planes, feat_dim] -> [bs, #planes, feat_dim, W] -> [bs x #planes, feat_dim, W] -> [bs x #planes, feat_dim, 1, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.permute(
                0, 2, 3, 1).reshape(
                    (bs * self.n_planes, -1, self.resolution)).unsqueeze(2)
            # [bs x #planes, feat_dim, 1, W] -> [bs x #planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(
                -1, -1, self.resolution, -1)

        return xyz_pos_enc_embeds

    def _add_y(self,
               bs,
               xyz_pos_encs,
               pos_enc_embed,
               w=None,
               fused_modconv=False):
        # [#planes, pos_enc_ch, H, 1], each plane has same Y coords
        selected_xyz_pos_encs = xyz_pos_encs[0, :, :, :, :1]
        # [#planes, pos_enc_ch, H, 1] -> [#planes, H, pos_enc_ch, 1] -> [#planes x H, pos_enc_ch]
        flat_xyz_pos_encs = selected_xyz_pos_encs.permute(0, 2, 1, 3).reshape(
            (self.n_planes * self.resolution, self.pos_enc_total_ch))
        if 'mlp' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes x H, feat_dim]
            flat_xyz_pos_enc_embeds = pos_enc_embed(flat_xyz_pos_encs)
        elif 'conv' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes x H, pos_enc_ch] -> [#planes x H, pos_enc_ch, 1, 1]
            flat_xyz_pos_encs = flat_xyz_pos_encs.unsqueeze(-1).unsqueeze(-1)
            # [#planes x H, feat_dim, 1, 1] -> [#planes x H, feat_dim]
            flat_xyz_pos_enc_embeds = pos_enc_embed(flat_xyz_pos_encs)[..., 0,
                                                                       0]
        elif 'modulated' in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes, pos_enc_ch, H, 1] -> [H, #planes, pos_enc_ch, 1] -> [H x #planes, pos_enc_ch]
            flat_xyz_pos_encs = selected_xyz_pos_encs.permute(
                2, 0, 1, 3).reshape(
                    (self.resolution * self.n_planes, self.pos_enc_total_ch))
            # [bs, H x #planes, pos_enc_ch] -> [bs x H x #planes, pos_enc_ch]
            flat_xyz_pos_encs = flat_xyz_pos_encs.unsqueeze(0).expand(
                bs, -1, -1)
            flat_xyz_pos_encs = flat_xyz_pos_encs.reshape(
                (bs * self.resolution * self.n_planes, self.pos_enc_total_ch))
            # [bs x H x #planes, pos_enc_ch] -> [bs x H x #planes, pos_enc_ch, 1, 1]
            flat_xyz_pos_encs = flat_xyz_pos_encs.unsqueeze(-1).unsqueeze(-1)
            # [bs, dim] -> [bs, 1, dim] -> [bs, H, dim] -> [bs x H, dim]
            w = w.unsqueeze(1).expand(-1, self.resolution, -1).reshape(
                (bs * self.resolution, -1))
            # [#planes x H, feat_dim, 1, 1] -> [#planes x H, feat_dim]
            flat_xyz_pos_enc_embeds = pos_enc_embed(
                flat_xyz_pos_encs,
                w,
                fused_modconv=fused_modconv,
                splitted=True,
                n_planes=self.n_planes)[..., 0, 0]
        else:
            raise ValueError

        if 'modulated' not in self.torgba_cond_on_pos_enc_embed_func:
            # [#planes x H, feat_dim] -> [#planes, H, feat_dim]
            xyz_pos_enc_embeds = flat_xyz_pos_enc_embeds.reshape(
                (self.n_planes, self.resolution, -1))
            # [#planes, H, feat_dim] -> [#planes, feat_dim, H] -> [#planes, feat_dim, H, 1]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.permute(0, 2,
                                                            1).unsqueeze(3)
            # [#planes, feat_dim, H, 1] -> [#planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(
                -1, -1, -1, self.resolution)
            # [#planes, feat_dim, H, W] -> [bs x #planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.repeat(bs, 1, 1, 1)
        else:
            # [bs x H x #planes, feat_dim] -> [bs, H, #planes, feat_dim]
            xyz_pos_enc_embeds = flat_xyz_pos_enc_embeds.reshape(
                (bs, self.resolution, self.n_planes, -1))
            # [bs, H, #planes, feat_dim] -> [bs, #planes, feat_dim, H] -> [bs x #planes, feat_dim, H, 1]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.permute(
                0, 2, 3, 1).reshape(
                    (bs * self.n_planes, -1, self.resolution)).unsqueeze(3)
            # [bs x #planes, feat_dim, H, 1] -> [bs x #planes, feat_dim, H, W]
            xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(
                -1, -1, -1, self.resolution)

        return xyz_pos_enc_embeds


#----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):

    def __init__(
            self,
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output image resolution.
            img_channels,  # Number of color channels.
            channel_base=32768,  # Overall multiplier for the number of channels.
            channel_max=512,  # Maximum number of channels in any layer.
            num_fp16_res=0,  # Use FP16 for the N highest resolutions.
            # MPI
        pos_enc_multires=0,  # Number of channels for positional encoding.
            torgba_cond_on_pos_enc='normalize_add_z',  # Whether to condition on Z or XYZ.
            torgba_cond_on_pos_enc_embed_func='modulated_lrelu',
            # how to produce MPI's RGB-a
            torgba_sep_background=False,  # Whether to generate background and foreground separately.
            build_background_from_rgb=False,  # Whether to build background image from boundaries of RGB.
            build_background_from_rgb_ratio=0.05,
            cond_on_pos_enc_only_alpha=False,  # Whether to only use "cond_on_pos_enc" for alpha channels.
            gen_alpha_largest_res=256,
            **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        # ensure it is exponential of 2
        assert img_resolution >= 4 and img_resolution & (img_resolution -
                                                         1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        # self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions
        }
        fp16_resolution = max(2**(self.img_resolution_log2 + 1 - num_fp16_res),
                              8)
        print('\n\nfp16_resolution: ', fp16_resolution, '\n\n')

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                pos_enc_multires=pos_enc_multires,
                torgba_cond_on_pos_enc=torgba_cond_on_pos_enc,
                torgba_cond_on_pos_enc_embed_func=
                torgba_cond_on_pos_enc_embed_func,
                torgba_sep_background=torgba_sep_background,
                build_background_from_rgb=build_background_from_rgb,
                build_background_from_rgb_ratio=build_background_from_rgb_ratio,
                cond_on_pos_enc_only_alpha=cond_on_pos_enc_only_alpha,
                gen_alpha_largest_res=gen_alpha_largest_res,
                **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)
            print(
                f'\n[G, res {res}] self.num_ws: conv {block.num_conv}, torgb {block.num_torgb}\n'
            )

        self.torgba_cond_on_pos_enc = torgba_cond_on_pos_enc
        self.torgba_cond_on_pos_enc_embed_func = torgba_cond_on_pos_enc_embed_func

    def forward(self,
                ws,
                xyz_coords=None,
                xyz_coords_only_z=False,
                enable_feat_net_grad=True,
                n_planes=32,
                **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(
                    ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            # NOTE: mpi_xyz_coords dict of xyz coords, each value is of shape [#planes, H, W, 3]
            if xyz_coords is not None:
                tmp_xyz_coords = xyz_coords[res]
            else:
                tmp_xyz_coords = None

            x, img = block(
                x,
                img,
                cur_ws,
                xyz_coords=tmp_xyz_coords,
                xyz_coords_only_z=xyz_coords_only_z,
                n_planes=n_planes,
                enable_feat_net_grad=enable_feat_net_grad,
                **block_kwargs)
        return img


#----------------------------------------------------------------------------


@persistence.persistent_class
class Generator(torch.nn.Module):

    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        # img_channels,                  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
        # MPI-related
        n_planes=1,
        plane_channels=4,
        pos_enc_multires=0,  # Number of channels for positional encoding.
        # ToRGBA-related
        torgba_cond_on_pos_enc='cond_xyz',  # Whether to condition on Z or XYZ.
        torgba_cond_on_pos_enc_embed_func='mlp',
        torgba_cond_on_pos_enc_learnable='none',
        background_alpha_full=False,
        # how to produce MPI's RGB-a
        torgba_sep_background=False,  # Whether to generate background and foreground separately.
        build_background_from_rgb=False,  # Whether to build background image from boundaries of RGB.
        build_background_from_rgb_ratio=0.05,
        cond_on_pos_enc_only_alpha=False,  # Whether to only use "cond_on_pos_enc" for alpha channels.
        gen_alpha_largest_res=256,
        G_final_img_act='none',
        depth2alpha_z_range=1.0,
        depth2alpha_n_z_bins=None,
    ):
        super().__init__()

        self.step = 0
        self.epoch = 0

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = 4

        self.background_alpha_full = background_alpha_full

        self.torgba_cond_on_pos_enc = torgba_cond_on_pos_enc

        self.G_final_img_act = G_final_img_act
        assert self.G_final_img_act in ['none', 'sigmoid',
                                        'tanh'], f'{self.G_final_img_act}'

        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=self.img_channels,
            pos_enc_multires=pos_enc_multires,
            torgba_cond_on_pos_enc=torgba_cond_on_pos_enc,
            torgba_cond_on_pos_enc_embed_func=torgba_cond_on_pos_enc_embed_func,
            torgba_sep_background=torgba_sep_background,
            build_background_from_rgb=build_background_from_rgb,
            build_background_from_rgb_ratio=build_background_from_rgb_ratio,
            cond_on_pos_enc_only_alpha=cond_on_pos_enc_only_alpha,
            gen_alpha_largest_res=gen_alpha_largest_res,
            **synthesis_kwargs)

        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.num_ws,
            **mapping_kwargs)

        print('\n[G] total self.num_ws: ', self.num_ws, '\n')

        self.conv_clamp = synthesis_kwargs['conv_clamp']

    def set_tune_toalpha(self):
        assert self.torgba_cond_on_pos_enc in [
            'add_z', 'normalize_add_z', 'add_xyz', 'normalize_add_xyz'
        ]
        for res in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f'b{res}')
            block.toalpha.requires_grad_(True)
            if 'xyz' in self.torgba_cond_on_pos_enc:
                block.pos_enc_embed_x.requires_grad_(True)
                block.pos_enc_embed_y.requires_grad_(True)
                block.pos_enc_embed_z.requires_grad_(True)
            else:
                block.pos_enc_embed.requires_grad_(True)

    def set_tune_tobackground(self):
        assert self.torgba_cond_on_pos_enc in [
            'add_z', 'normalize_add_z', 'add_xyz', 'normalize_add_xyz'
        ]
        for res in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f'b{res}')
            block.toalpha.requires_grad_(True)
            block.tobackground.requires_grad_(True)

    def synthesize(self,
                   *,
                   ws=None,
                   n_planes=32,
                   mpi_xyz_coords=None,
                   xyz_coords_only_z=False,
                   enable_syn_feat_net_grad=True,
                   **synthesis_kwargs):

        img = self.synthesis(
            ws,
            xyz_coords=mpi_xyz_coords,
            enable_feat_net_grad=enable_syn_feat_net_grad,
            xyz_coords_only_z=xyz_coords_only_z,
            n_planes=n_planes,
            **synthesis_kwargs)

        if self.G_final_img_act == 'none':
            # [-1, 1] -> [0, 1]
            img = (torch.clamp(img, min=-1.0, max=1.0) + 1.0) / 2.0
        elif self.G_final_img_act == 'sigmoid':
            img = torch.sigmoid(img)
        elif self.G_final_img_act == 'tanh':
            img = (torch.tanh(img) + 1.0) / 2.0
        else:
            raise ValueError

        if self.background_alpha_full:
            bs = img.shape[0]
            full_alpha = torch.ones(
                (bs, 1, self.img_resolution, self.img_resolution),
                device=img.device)
            img = torch.cat((img[:, :-1, ...], full_alpha), dim=1)

        # [B, #planes x 4, H, W] -> [B, #planes, 4, tex_h, tex_w]
        img = img.reshape((img.shape[0], n_planes, 4, self.img_resolution,
                           self.img_resolution))

        return img

    def forward(self,
                z,
                c,
                mpi_xyz_coords,
                xyz_coords_only_z,
                n_planes,
                z_interpolation_ws=None,
                truncation_psi=1,
                truncation_cutoff=None,
                enable_mapping_grad=True,
                enable_syn_feat_net_grad=True,
                **synthesis_kwargs):

        with torch.set_grad_enabled(enable_mapping_grad):
            ws = self.mapping(
                z,
                c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff)

        img = self.synthesize(
            ws=ws,
            n_planes=n_planes,
            mpi_xyz_coords=mpi_xyz_coords,
            xyz_coords_only_z=xyz_coords_only_z,
            enable_syn_feat_net_grad=enable_syn_feat_net_grad,
            **synthesis_kwargs)

        return img

    def set_device(self, device):
        self.device = device


#----------------------------------------------------------------------------


@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):

    def __init__(
            self,
            in_channels,  # Number of input channels, 0 = first block.
            tmp_channels,  # Number of intermediate channels.
            out_channels,  # Number of output channels.
            resolution,  # Resolution of this block.
            img_channels,  # Number of input color channels.
            first_layer_idx,  # Index of the first layer.
            architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            resample_filter=[
                1, 3, 3, 1
            ],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            use_fp16=False,  # Use FP16 for this block?
            fp16_channels_last=False,  # Use channels-last memory format with FP16?
            freeze_layers=0,  # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(
                img,
                [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(
                img,
                self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


#----------------------------------------------------------------------------


@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):

    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(
        ):  # as_tensor results are registered as constants
            G = torch.min(
                torch.as_tensor(self.group_size),
                torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3,
                        4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y],
                      dim=1)  # [NCHW]   Append to input as new channels.
        return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):

    def __init__(
        self,
        in_channels,  # Number of input channels.
        cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_mbstd_in_D=True,
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        self.use_mbstd_in_D = use_mbstd_in_D

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(
                img_channels,
                in_channels,
                kernel_size=1,
                activation=activation)
        self.mbstd = MinibatchStdLayer(
            group_size=mbstd_group_size, num_channels=mbstd_num_channels
        ) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=activation,
            conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(
            in_channels * (resolution**2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels,
                                       1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [
            None, self.in_channels, self.resolution, self.resolution
        ])  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(
                img,
                [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.use_mbstd_in_D:
            if self.mbstd is not None:
                x = self.mbstd(x)
        else:
            bs, _, h, w = x.shape
            placeholder = torch.zeros((bs, 1, h, w), device=x.device)
            x = torch.cat((x, placeholder), dim=1)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(
                dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


#----------------------------------------------------------------------------


@persistence.persistent_class
class Discriminator(torch.nn.Module):

    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        use_mbstd_in_D=True,
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
        D_stylegan2_ori_mapping=False,
    ):
        super().__init__()

        self.epoch = 0
        self.step = 0

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(2**(self.img_resolution_log2 + 1 - num_fp16_res),
                              8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(
            img_channels=img_channels,
            architecture=architecture,
            conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        self.D_stylegan2_ori_mapping = D_stylegan2_ori_mapping
        if c_dim > 0:
            if self.D_stylegan2_ori_mapping:
                raise NotImplementedError
            else:
                self.mapping = torch.nn.Linear(c_dim, cmap_dim)

        self.b4 = DiscriminatorEpilogue(
            channels_dict[4],
            cmap_dim=cmap_dim,
            resolution=4,
            use_mbstd_in_D=use_mbstd_in_D,
            **epilogue_kwargs,
            **common_kwargs)

    def forward(self,
                img,
                alpha_placeholder,
                flat_pose,
                block_kwargs={},
                **kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        pose_embed = None
        if self.c_dim > 0:
            if self.D_stylegan2_ori_mapping:
                raise NotImplementedError
                # pose_embed = self.mapping(None, flat_pose)
            else:
                pose_embed = self.mapping(flat_pose)
                pose_embed = normalize_2nd_moment(pose_embed)

        x = self.b4(x, img, pose_embed)

        latent = None
        position = None

        return x, latent, position


#----------------------------------------------------------------------------
