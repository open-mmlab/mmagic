# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from ..stylegan3.stylegan3_modules import FullyConnectedLayer
from ..stylegan3.stylegan3_ops.ops import bias_act, upfirdn2d
from .gmpi_utils import conv2d_resample, fma

FLOATING_EPS = 1e-8


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

    def embed(x):
        return embedder_obj.embed(x)

    return embed, embedder_obj.out_dim


def compute_pos_enc(xyz_coords, pos_enc_fn, tex_h, tex_w, only_z=False):

    if only_z:
        n_coords = 1
        tex_h = 1
        tex_w = 1
    else:
        n_coords = 3

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


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulated_conv2d(
    x,
    # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,
    # Weight tensor of shape
    # [out_channels, in_channels, kernel_height, kernel_width].
    styles,
    # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,
    # Optional noise tensor to add to the output activations.
    up=1,
    # Integer upsampling factor.
    down=1,
    # Integer downsampling factor.
    padding=0,
    # Padding with respect to the upsampled image.
    resample_filter=None,
    # Low-pass filter to apply when resampling activations. Must be
    # prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,
    # Apply weight demodulation?
    flip_weight=True,
    # False = convolution, True = correlation
    # (matches torch.nn.functional.conv2d).
    fused_modconv=True,
    # Perform modulation, convolution, and demodulation as a single
    # fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    # misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    # misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    # misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

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
        x = conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma(x,
                    dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1),
                    noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    # misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample(
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


# @persistence.persistent_class
class MappingNetwork(torch.nn.Module):

    def __init__(
        self,
        z_dim,
        # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,
        # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,
        # Intermediate latent (W) dimensionality.
        num_ws,
        # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,
        # Number of mapping layers.
        embed_features=None,
        # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,
        # Number of intermediate features in the mapping layers,
        # None = same as w_dim.
        activation='lrelu',
        # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,
        # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,
        # Decay for tracking the moving average of W during training,
        # None = do not track.
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
                # misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                # misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training:
            if not skip_w_avg_update:
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


# @persistence.persistent_class
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
            conv_clamp=None,
            # Clamp the output of convolution layers to +-X,
            # None = disable clamping.
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
        if channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
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
        # in_resolution = self.resolution // self.up
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
        act_clamp = self.conv_clamp * gain
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp)
        return x


# @persistence.persistent_class
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
        if channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size,
                         kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w, fused_modconv=True, split=False, n_planes=1):
        bs = w.shape[0]
        styles = self.affine(w) * self.weight_gain
        if split:
            # We make generation plane-specific.
            # Therefore, we need to duplicate styles tensor.
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


# @persistence.persistent_class
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
        if channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format

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

    def forward(self, x, w, fused_modconv=True, split=False, n_planes=1):
        bs = w.shape[0]
        styles1 = self.affine1(w) * self.weight_gain1
        if split:
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
        if split:
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
        if split:
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
        if split:
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


# @persistence.persistent_class
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
            conv_clamp=None,
            # Clamp the output of convolution layers to +-X,
            # None = disable clamping.
            use_fp16=False,  # Use FP16 for this block?
            fp16_channels_last=False,
            # Use channels-last memory format with FP16?
            pos_enc_multires=0,
            # Number of channels for positional encoding.
            torgba_sep_background=False,
            # Whether to generate background and foreground separately.
            build_background_from_rgb=False,
            # Whether to build background image from boundaries of RGB.
            background_from_rgb_ratio=0.05,
            cond_on_pos_enc_only_alpha=False,
            # Whether to only use "cond_on_pos_enc" for alpha channels.
            gen_alpha_largest_res=256,
            # Largest resolution to generate alpha maps.
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

        self.torgba_sep_background = torgba_sep_background
        self.build_background_from_rgb = build_background_from_rgb
        self.background_from_rgb_ratio = background_from_rgb_ratio
        self.cond_on_pos_enc_only_alpha = cond_on_pos_enc_only_alpha
        self.gen_alpha_largest_res = gen_alpha_largest_res

        self.gen_alpha_this_res = self.gen_alpha_largest_res >= self.resolution

        if self.build_background_from_rgb:
            assert self.torgba_sep_background

        self.pos_enc_fn, pos_enc_ch_single_embed = get_embedder(
            pos_enc_multires, 1, use_embed=True)

        self.pos_enc_total_ch = pos_enc_ch_single_embed

        print('\nconv_clamp: ', conv_clamp, '\n')

        if self.gen_alpha_this_res:
            intermediate_ch_list = [
                out_channels // 4, out_channels // 2, out_channels
            ]
            self.pos_enc_embed = ToRGBLayerDeeperModulatedConv(
                self.pos_enc_total_ch,
                out_channels,
                intermediate_channels=intermediate_ch_list,
                w_dim=w_dim,
                act_name='lrelu',
                conv_clamp=conv_clamp,
                channels_last=self.channels_last)

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

            extra_channels = 0

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

            self.num_torgb += 1

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

            w_iter = iter(ws.unbind(dim=1))
            if self.use_fp16 and not force_fp32:
                dtype = torch.float16
            else:
                dtype = torch.float32
            if self.channels_last:
                memory_format = torch.channels_last
            else:
                memory_format = torch.contiguous_format
            if fused_modconv is None:
                fused_modconv = (not self.training) and (
                    dtype == torch.float32 or int(x.shape[0]) == 1)

            # Input.
            if self.in_channels == 0:
                x = self.const.to(dtype=dtype, memory_format=memory_format)
                x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
            else:
                x = x.to(dtype=dtype, memory_format=memory_format)

            # Main layers.
            if self.in_channels == 0:
                w_conv1 = next(w_iter)
                x = self.conv1(
                    x, w_conv1, fused_modconv=fused_modconv, **layer_kwargs)
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
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.gen_alpha_this_res:
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
            xyz_pos_encs = xyz_pos_encs[:, :, 2, ...]

            xyz_pos_enc_embeds = self._add_z(
                bs,
                xyz_pos_encs,
                self.pos_enc_embed,
                w=w_conv1,
                fused_modconv=fused_modconv)

            cond_x = x

            # [bs, feat_dim, 1, 1]
            cond_x_mean, cond_x_std = calc_mean_std(cond_x)
            cond_x = (cond_x - cond_x_mean) / (cond_x_std + FLOATING_EPS)

            # [bs, 1, feat_dim, H, W] -> [bs, #planes, feat_dim, H, W]
            # -> [bs x #planes, feat_dim, H, W]
            cond_x = cond_x.unsqueeze(1).expand(-1, self.n_planes, -1, -1,
                                                -1).reshape(
                                                    (bs * self.n_planes, -1,
                                                     self.resolution,
                                                     self.resolution))

            cond_x = cond_x + xyz_pos_enc_embeds
        else:
            cond_x = None

        # NOTE: we use same w for all toRGB layers
        w_for_rgba = next(w_iter)

        w_rgb = w_for_rgba
        w_alpha = w_for_rgba

        with torch.no_grad():
            if self.resolution >= 0:
                # NOTE: we use elements from left/top/right boundaries.
                # We do not use elements near bottom boundary as there are
                # necks.
                cur_background_x = torch.zeros(
                    x.shape, device=x.device, dtype=dtype, requires_grad=False)
                tmp_pad = max(
                    1,
                    int(
                        np.floor(self.background_from_rgb_ratio *
                                 self.resolution)))
                # left
                cur_background_x[:, :, :, :tmp_pad] = x[:, :, :, :
                                                        tmp_pad].detach()
                # right
                cur_background_x[:, :, :, -tmp_pad:] = x[:, :, :,
                                                         -tmp_pad:].detach()
                # # top
                # cur_background_x[:, :, :tmp_pad, :] =
                # x[:, :, :tmp_pad, :].detach()
                tmp_top = 0
                # interpolation
                tmp_start_col = tmp_pad
                tmp_end_col = self.resolution - tmp_pad
                if tmp_start_col < tmp_end_col:
                    tmp_cols = torch.arange(
                        tmp_start_col, tmp_end_col, device=x.device).reshape(
                            (1, 1, 1, -1))
                    tmp_right_ratios = (tmp_cols - tmp_start_col) / (
                        tmp_end_col - tmp_start_col + FLOATING_EPS)
                    # [bs, feat_dim, #rows, #columns]
                    tmp_left_feat = x[:, :, tmp_top:,
                                      tmp_start_col:(tmp_start_col +
                                                     1)].detach()
                    tmp_right_feat = x[:, :, tmp_top:,
                                       (tmp_end_col - 1):tmp_end_col].detach()
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
            split=False,
            n_planes=self.n_planes)

        cur_background = cur_background.unsqueeze(1)

        # [bs, 3, H, W]
        # w_rgb = next(w_iter)
        single_rgb = self.torgb(
            x,
            w_rgb,
            fused_modconv=fused_modconv,
            split=False,
            n_planes=self.n_planes)

        # print("single_rgb: ", single_rgb.shape, single_rgb.dtype, "\n\n")

        # [bs, 3, H, W] -> [bs, 1, 3, H, W] -> [bs, #planes - 1, 3, H, W]
        cur_rgb = single_rgb.unsqueeze(1).expand(-1, self.n_planes - 1, -1, -1,
                                                 -1)
        # [bs, #planes, 3, H, W]
        cur_rgb = torch.cat((cur_rgb, cur_background), dim=1)

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
                split=True,
                n_planes=self.n_planes)
        else:
            # NOTE: we do not generate alpha maps at this resolution
            cur_alpha = torch.zeros(
                (bs * self.n_planes, 1, self.resolution, self.resolution),
                device=cur_rgb.device)

        # [bs x #planes, 4, H, W]
        y = torch.cat((cur_rgb, cur_alpha), dim=1)

        y = y.reshape(
            (bs, self.n_planes, 4, self.resolution, self.resolution)).view(
                (bs, self.n_planes * 4, self.resolution, self.resolution))
        # [B, #planes, 4, H, W] -> [B, #planes * 4, H, W]
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
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
        # [#planes, pos_enc_ch, 1, 1], since Z is same for whole plane
        # we only need one elements
        selected_xyz_pos_encs = xyz_pos_encs[0, :, :, :1, :1]

        # [bs, #planes, pos_enc_ch, 1, 1]
        # -> [bs x #planes, pos_enc_ch, 1, 1]
        selected_xyz_pos_encs = selected_xyz_pos_encs.unsqueeze(0).expand(
            bs, -1, -1, -1, -1)
        selected_xyz_pos_encs = selected_xyz_pos_encs.reshape(
            (bs * self.n_planes, self.pos_enc_total_ch, 1, 1))
        xyz_pos_enc_embeds = pos_enc_embed(
            selected_xyz_pos_encs,
            w,
            fused_modconv=fused_modconv,
            split=True,
            n_planes=self.n_planes)

        # [bs x #planes, feat_dim, 1, 1] -> [bs x #planes, feat_dim, H, W]
        xyz_pos_enc_embeds = xyz_pos_enc_embeds.expand(-1, -1, self.resolution,
                                                       self.resolution)

        return xyz_pos_enc_embeds


# @persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):

    def __init__(
            self,
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output image resolution.
            img_channels,  # Number of color channels.
            channel_base=32768,
            # Overall multiplier for the number of channels.
            channel_max=512,  # Maximum number of channels in any layer.
            num_fp16_res=0,  # Use FP16 for the N highest resolutions.
            pos_enc_multires=0,
            # Number of channels for positional encoding.
            torgba_sep_background=False,
            # Whether to generate background and foreground separately.
            build_background_from_rgb=False,
            # Whether to build background image from boundaries of RGB.
            background_from_rgb_ratio=0.05,
            cond_on_pos_enc_only_alpha=False,
            # Whether to only use "cond_on_pos_enc" for alpha channels.
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
                torgba_sep_background=torgba_sep_background,
                build_background_from_rgb=build_background_from_rgb,
                background_from_rgb_ratio=background_from_rgb_ratio,
                cond_on_pos_enc_only_alpha=cond_on_pos_enc_only_alpha,
                gen_alpha_largest_res=gen_alpha_largest_res,
                **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)
            print(f'\n[G, res {res}] self.num_ws: conv {block.num_conv},\n'
                  f'torgb {block.num_torgb}\n')

    def forward(self,
                ws,
                xyz_coords=None,
                xyz_coords_only_z=False,
                enable_feat_net_grad=True,
                n_planes=32,
                **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            # misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
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
            # NOTE: mpi_xyz_coords dict of xyz coords,
            # each value is of shape [#planes, H, W, 3]
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
