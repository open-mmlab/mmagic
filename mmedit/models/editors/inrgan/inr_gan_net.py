import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmcv.ops.fused_bias_leakyrelu import fused_bias_leakyrelu
from mmengine.model import BaseModule
from mmedit.registry import MODELS
from ..stylegan2 import ModulatedToRGB
from utils import conv2d_resample

from typing import Any, List, Tuple, Union

# TODO: Code Refactoring.

def maybe_upsample(x, upsampling_mode: str=None, up: int=1) -> Tensor:
    if up == 1 or upsampling_mode is None:
        return x

    if upsampling_mode == 'bilinear':
        x = F.interpolate(x, mode='bilinear', align_corners=True, scale_factor=up)
    else:
        x = F.interpolate(x, mode=upsampling_mode, scale_factor=up)

    return x

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

def generate_horizontal_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [0.0, 1.0], 4.0)


def generate_vertical_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0, 0.0], 4.0)


def generate_diag_main_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_anti_diag_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_wavefront_basis(num_feats: int, basis_block: List[float], period_length: float) -> Tensor:
    period_coef = 2.0 * np.pi / period_length
    basis = torch.tensor([basis_block]).repeat(num_feats, 1) # [num_feats, 2]
    powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1) # [num_feats, 1]
    result = basis * powers * period_coef # [num_feats, 2]

    return result.float()


def generate_logarithmic_basis(
    resolution: int,
    max_num_feats: int=np.float('inf'),
    remove_lowest_freq: bool=False,
    use_diagonal: bool=True) -> Tensor:
    """
    Generates a directional logarithmic basis with the following directions:
        - horizontal
        - vertical
        - main diagonal
        - anti-diagonal
    """
    max_num_feats_per_direction = np.ceil(np.log2(resolution)).astype(int)
    bases = [
        generate_horizontal_basis(max_num_feats_per_direction),
        generate_vertical_basis(max_num_feats_per_direction),
    ]

    if use_diagonal:
        bases.extend([
            generate_diag_main_basis(max_num_feats_per_direction),
            generate_anti_diag_basis(max_num_feats_per_direction),
        ])

    if remove_lowest_freq:
        bases = [b[1:] for b in bases]

    # If we do not fit into `max_num_feats`, then trying to remove the features in the order:
    # 1) anti-diagonal 2) main-diagonal
    # while (max_num_feats_per_direction * len(bases) > max_num_feats) and (len(bases) > 2):
    #     bases = bases[:-1]

    basis = torch.cat(bases, dim=0)

    # If we still do not fit, then let's remove each second feature,
    # then each third, each forth and so on
    # We cannot drop the whole horizontal or vertical direction since otherwise
    # model won't be able to locate the position
    # (unless the previously computed embeddings encode the position)
    # while basis.shape[0] > max_num_feats:
    #     num_exceeding_feats = basis.shape[0] - max_num_feats
    #     basis = basis[::2]

    assert basis.shape[0] <= max_num_feats, \
        f"num_coord_feats > max_num_fixed_coord_feats: {basis.shape, max_num_feats}."

    return basis


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
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
            # x = bias_act.bias_act(x, b, act=self.activation)
            x = fused_bias_leakyrelu(x, b, negative_slope=0.2)
        return x

def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    # misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    # misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    # misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    # with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        # batch_size = int(batch_size)
    batch_size = int(batch_size)

    # misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool=False) -> Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[0, 0] = (-1, -1)
    - upper right corner: coords[img_size - 1, img_size - 1] = (1, 1)
    """
    if align_corners:   # 跳过
        row = torch.linspace(-1, 1, img_size, device=device).float() # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t().flip(dims=(0,)) # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1) # [batch_size, 2, img_size, img_size]

    return coords

class CoordFuser(nn.Module):
    """
    CoordFuser which concatenates coordinates across dim=1 (we assume channel_first format)
    """
    def __init__(self, cfg: DictConfig, w_dim: int, resolution: int):
        super().__init__()

        self.cfg = cfg
        self.resolution = resolution
        self.res_cfg = self.cfg.res_configs[str(resolution)]
        self.log_emb_size = self.res_cfg.get('log_emb_size', 0)
        self.random_emb_size = self.res_cfg.get('random_emb_size', 0)
        self.shared_emb_size = self.res_cfg.get('shared_emb_size', 0)
        self.predictable_emb_size = self.res_cfg.get('predictable_emb_size', 0)
        self.const_emb_size = self.res_cfg.get('const_emb_size', 0)
        self.fourier_scale = self.res_cfg.get('fourier_scale', np.sqrt(10))
        self.use_cosine = self.res_cfg.get('use_cosine', False)
        self.use_raw_coords = self.res_cfg.get('use_raw_coords', False)
        self.init_dist = self.res_cfg.get('init_dist', 'randn')
        self._fourier_embs_cache = None
        self._full_cache = None
        self.use_full_cache = cfg.get('use_full_cache', False)

        if self.log_emb_size > 0:
            self.register_buffer('log_basis', generate_logarithmic_basis(
                resolution, self.log_emb_size, use_diagonal=self.res_cfg.get('use_diagonal', False))) # [log_emb_size, 2]

        if self.random_emb_size > 0:
            self.register_buffer('random_basis', self.sample_w_matrix((self.random_emb_size, 2), self.fourier_scale))

        if self.shared_emb_size > 0:
            self.shared_basis = nn.Parameter(self.sample_w_matrix((self.shared_emb_size, 2), self.fourier_scale))

        if self.predictable_emb_size > 0:
            self.W_size = self.predictable_emb_size * self.cfg.coord_dim
            self.b_size = self.predictable_emb_size
            self.affine = FullyConnectedLayer(w_dim, self.W_size + self.b_size, bias_init=0)

        if self.const_emb_size > 0:
            self.const_embs = nn.Parameter(torch.randn(1, self.const_emb_size, resolution, resolution).contiguous())

        self.total_dim = self.get_total_dim()
        self.is_modulated = (self.predictable_emb_size > 0)

    def sample_w_matrix(self, shape: Tuple[int], scale: float):
        if self.init_dist == 'randn':
            return torch.randn(shape) * scale
        elif self.init_dist == 'rand':
            return (torch.rand(shape) * 2 - 1) * scale
        else:
            raise NotImplementedError(f"Unknown init dist: {self.init_dist}")

    def get_total_dim(self) -> int:
        if self.cfg.fallback:
            return 0

        total_dim = 0
        total_dim += (self.cfg.coord_dim if self.use_raw_coords else 0)
        if self.log_emb_size > 0:
            total_dim += self.log_basis.shape[0] * (2 if self.use_cosine else 1)
        total_dim += self.random_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.shared_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.predictable_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.const_emb_size

        return total_dim

    def forward(self, x: Tensor, w: Tensor=None, dtype=None, memory_format=None) -> Tensor:
        """
        Dims:
            @arg x is [batch_size, in_channels, img_size, img_size]
            @arg w is [batch_size, w_dim]
            @return out is [batch_size, in_channels + fourier_dim + cips_dim, img_size, img_size]
        """
        assert memory_format is torch.contiguous_format

        if self.cfg.fallback:
            return x

        batch_size, in_channels, img_size = x.shape[:3]
        out = x

        if self.use_full_cache and (not self._full_cache is None) and (self._full_cache.device == x.device) and \
           (self._full_cache.shape == (batch_size, self.get_total_dim(), img_size, img_size)):
           return torch.cat([x, self._full_cache], dim=1)

        if (not self._fourier_embs_cache is None) and (self._fourier_embs_cache.device == x.device) and \
           (self._fourier_embs_cache.shape == (batch_size, self.get_total_dim() - self.const_emb_size, img_size, img_size)):
            out = torch.cat([out, self._fourier_embs_cache], dim=1)
        else:
            raw_embs = []
            raw_coords = generate_coords(batch_size, img_size, x.device) # [batch_size, coord_dim, img_size, img_size]

            if self.use_raw_coords:
                out = torch.cat([out, raw_coords], dim=1)

            if self.log_emb_size > 0:
                log_bases = self.log_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, log_emb_size, 2]
                raw_log_embs = torch.einsum('bdc,bcxy->bdxy', log_bases, raw_coords) # [batch_size, log_emb_size, img_size, img_size]
                raw_embs.append(raw_log_embs)

            if self.random_emb_size > 0:
                random_bases = self.random_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, random_emb_size, 2]
                raw_random_embs = torch.einsum('bdc,bcxy->bdxy', random_bases, raw_coords) # [batch_size, random_emb_size, img_size, img_size]
                raw_embs.append(raw_random_embs)

            if self.shared_emb_size > 0:
                shared_bases = self.shared_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, shared_emb_size, 2]
                raw_shared_embs = torch.einsum('bdc,bcxy->bdxy', shared_bases, raw_coords) # [batch_size, shared_emb_size, img_size, img_size]
                raw_embs.append(raw_shared_embs)

            if self.predictable_emb_size > 0:
                # misc.assert_shape(w, [batch_size, None])
                mod = self.affine(w) # [batch_size, W_size + b_size]
                W = self.fourier_scale * mod[:, :self.W_size] # [batch_size, W_size]
                W = W.view(batch_size, self.predictable_emb_size, self.cfg.coord_dim) # [batch_size, predictable_emb_size, coord_dim]
                bias = mod[:, self.W_size:].view(batch_size, self.predictable_emb_size, 1, 1) # [batch_size, predictable_emb_size, 1]
                raw_predictable_embs = (torch.einsum('bdc,bcxy->bdxy', W, raw_coords) + bias) # [batch_size, predictable_emb_size, img_size, img_size]
                raw_embs.append(raw_predictable_embs)

            if len(raw_embs) > 0:
                raw_embs = torch.cat(raw_embs, dim=1) # [batch_suze, log_emb_size + random_emb_size + predictable_emb_size, img_size, img_size]
                raw_embs = raw_embs.contiguous() # [batch_suze, -1, img_size, img_size]
                out = torch.cat([out, raw_embs.sin().to(dtype=dtype, memory_format=memory_format)], dim=1) # [batch_size, -1, img_size, img_size]

                if self.use_cosine:
                    out = torch.cat([out, raw_embs.cos().to(dtype=dtype, memory_format=memory_format)], dim=1) # [batch_size, -1, img_size, img_size]

        if self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > x.shape[1]:
            self._fourier_embs_cache = out[:, x.shape[1]:].detach()

        if self.const_emb_size > 0:
            const_embs = self.const_embs.repeat([batch_size, 1, 1, 1])
            const_embs = const_embs.to(dtype=dtype, memory_format=memory_format)
            out = torch.cat([out, const_embs], dim=1) # [batch_size, total_dim, img_size, img_size]

        if self.use_full_cache and self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > x.shape[1]:
            self._full_cache = out[:, x.shape[1]:].detach()

        return out


class CoordsInput(nn.Module):
    def __init__(self, cfg: DictConfig, w_dim: int):
        super().__init__()

        self.cfg = cfg
        self.coord_fuser = CoordFuser(self.cfg.coord_fuser_cfg, w_dim, self.cfg.resolution)

    def get_total_dim(self) -> int:
        return self.coord_fuser.total_dim

    def forward(self, batch_size: int, w: Optional[Tensor]=None, device='cpu', dtype=None, memory_format=None) -> Tensor:
        dummy_input = torch.empty(batch_size, 0, self.cfg.resolution, self.cfg.resolution)
        dummy_input = dummy_input.to(device, dtype=dtype, memory_format=memory_format)
        out = self.coord_fuser(dummy_input, w, dtype=dtype, memory_format=memory_format)

        return out


class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        # x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        x = fused_bias_leakyrelu(x, self.bias.to(x.dtype))
        return x


class GenInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, w_dim: int):
        super().__init__()

        self.cfg = cfg

        if self.cfg.type == 'const':
            self.input = torch.nn.Parameter(torch.randn([channel_dim, self.cfg.resolution, self.cfg.resolution]))
            self.total_dim = channel_dim
        elif self.cfg.type == 'coords':
            self.input = CoordsInput(self.cfg, w_dim)
            self.total_dim = self.input.get_total_dim()
        else:
            raise NotImplementedError

    def forward(self, batch_size: int, w: Tensor=None, device=None, dtype=None, memory_format=None) -> Tensor:
        if self.cfg.type == 'const':
            x = self.input.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        elif self.cfg.type == 'coords':
            x = self.input(batch_size, w, device=device, dtype=dtype, memory_format=memory_format)
        else:
            raise NotImplementedError

        return x


class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
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
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
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
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

def fmm_modulate_linear(x: Tensor, weight: Tensor, styles: Tensor, noise=None, activation: str="demod") -> Tensor:
    """
    x: [batch_size, c_in, height, width]
    weight: [c_out, c_in, 1, 1]
    style: [batch_size, num_mod_params]
    noise: Optional[batch_size, 1, height, width]
    """
    batch_size, c_in, h, w = x.shape
    c_out, c_in, kh, kw = weight.shape
    rank = styles.shape[1] // (c_in + c_out)

    assert kh == 1 and kw == 1
    assert styles.shape[1] % (c_in + c_out) == 0

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = styles[:, :c_out * rank] # [batch_size, left_matrix_size]
    right_matrix = styles[:, c_out * rank:] # [batch_size, right_matrix_size]

    left_matrix = left_matrix.view(batch_size, c_out, rank) # [batch_size, c_out, rank]
    right_matrix = right_matrix.view(batch_size, rank, c_in) # [batch_size, rank, c_in]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank) # [batch_size, c_out, c_in]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight.squeeze(3).squeeze(2).unsqueeze(0) * (modulation + 1.0) # [batch_size, c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, c_out, c_in]
    W = W.to(dtype=x.dtype)

    # out = torch.einsum('boi,bihw->bohw', W, x)
    x = x.view(batch_size, c_in, h * w) # [batch_size, c_in, h * w]
    out = torch.bmm(W, x) # [batch_size, c_out, h * w]
    out = out.view(batch_size, c_out, h, w) # [batch_size, c_out, h, w]

    if not noise is None:
        out = out.add_(noise)

    return out

class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
        instance_norm   = False,        # Use instance norm?
        cfg             = {},           # Additional config
    ):
        super().__init__()

        self.cfg = OmegaConf.create(cfg)
        self.resolution = resolution
        self.up = up
        self.activation = activation
        self.conv_clamp = conv_clamp
        # self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        if self.cfg.fmm.enabled:
            self.affine = FullyConnectedLayer(w_dim, (in_channels + out_channels) * self.cfg.fmm.rank, bias_init=0)
        else:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if self.cfg.use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.instance_norm = instance_norm

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        # misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.cfg.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.cfg.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster

        if self.instance_norm:
            x = x / (x.std(dim=[2,3], keepdim=True) + 1e-8) # [batch_size, c, h, w]

        if self.cfg.fmm.enabled:
            x = fmm_modulate_linear(x=x, weight=self.weight, styles=styles, noise=noise, activation=self.cfg.fmm.activation)
        else:
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
                padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight,
                fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        # x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        x = fused_bias_leakyrelu(x, self.bias.to(x.dtype))
        return x


class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        cfg                 = {},           # Additional config
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = OmegaConf.create(cfg)
        self.in_channels = in_channels
        self.w_dim = w_dim

        if resolution <= self.cfg.input.resolution:
            self.resolution = self.cfg.input.resolution
            self.up = 1
            self.input_resolution = self.cfg.input.resolution
        else:
            self.resolution = resolution
            self.up = 2
            self.input_resolution = resolution // 2

        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        # self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        kernel_size = self.cfg.coords.kernel_size if self.cfg.coords.enabled else 3

        if in_channels == 0:
            self.input = GenInput(self.cfg.input, out_channels, w_dim)
            conv1_in_channels = self.input.total_dim
        else:
            if self.cfg.coords.enabled and (not self.cfg.coords.per_resolution or self.resolution > self.input_resolution):
                assert self.architecture != 'resnet'
                self.coord_fuser = CoordFuser(self.cfg.coords, self.w_dim, self.resolution)
                conv0_in_channels = in_channels + self.coord_fuser.total_dim
            else:
                self.coord_fuser = None
                conv0_in_channels = in_channels

            up_for_conv0 = self.up if self.cfg.upsampling_mode is None else 1
            self.conv0 = SynthesisLayer(conv0_in_channels, out_channels, w_dim=w_dim, resolution=self.resolution, up=up_for_conv0,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last,
                kernel_size=kernel_size, cfg=cfg, **layer_kwargs)
            self.num_conv += 1
            conv1_in_channels = out_channels

        self.conv1 = SynthesisLayer(conv1_in_channels, out_channels, w_dim=w_dim, resolution=self.resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, kernel_size=kernel_size, cfg=cfg,
            instance_norm=(in_channels > 0 and cfg.get('fmm', {}).get('instance_norm', False)), **layer_kwargs)
        self.num_conv += 1

        if self.cfg.get('num_extra_convs', {}).get(str(self.resolution), 0) > 0:
            assert self.architecture != 'resnet', "Not implemented for resnet"
            self.extra_convs = nn.ModuleList([
                SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=self.resolution,
                    conv_clamp=conv_clamp, channels_last=self.channels_last, kernel_size=kernel_size,
                    instance_norm=cfg.get('fmm', {}).get('instance_norm', False), cfg=cfg, **layer_kwargs)
                    for _ in range(self.cfg.num_extra_convs[str(self.resolution)])])
            self.num_conv += len(self.extra_convs)
        else:
            self.extra_convs = None

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            # self.torgb = ModulatedToRGB(in_channels=out_channels, style_channels=w_dim, out_channels=img_channels, upsample=False)
            self.num_torgb += 1

        # if in_channels != 0 and architecture == 'resnet': # 不需要
        #     self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=self.up,
        #         resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        # misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        if fused_modconv is None:
            # with misc.suppress_tracer_warnings(): # this value will be treated as a constant
            #     fused_modconv = (not self.training) and (dtype == torch.float32 or (isinstance(x, Tensor) and int(x.shape[0]) == 1))
            fused_modconv = (not self.training) and (dtype == torch.float32 or (isinstance(x, Tensor) and int(x.shape[0]) == 1))

        # Input.
        if self.in_channels == 0:
            conv1_w = next(w_iter)
            x = self.input(ws.shape[0], conv1_w, device=ws.device, dtype=dtype, memory_format=memory_format)
        else:
            # misc.assert_shape(x, [None, self.in_channels, self.input_resolution, self.input_resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        x = maybe_upsample(x, self.cfg.upsampling_mode, self.up)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, conv1_w, fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            conv0_w = next(w_iter)

            if self.coord_fuser is not None:
                x = self.coord_fuser(x, conv0_w, dtype=dtype, memory_format=memory_format)

            x = self.conv0(x, conv0_w, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        if not self.extra_convs is None:
            for conv, w in zip(self.extra_convs, w_iter):
                x = conv(x, w, fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            # misc.assert_shape(img, [None, self.img_channels, self.input_resolution, self.input_resolution])

            if self.up == 2:
                if self.cfg.upsampling_mode is None:
                    img = upfirdn2d.upsample2d(img, self.resample_filter)
                else:
                    img = maybe_upsample(img, self.cfg.upsampling_mode, 2)

        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        cfg             = {},       # Additional config
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()

        self.w_dim = w_dim
        self.cfg = OmegaConf.create(cfg)
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, cfg=cfg, **block_kwargs)
            self.num_ws += block.num_conv

            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            # misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


class Generator(torch.nn.Module):
    # TODO: add docstring.
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        cfg                 = {},   # Config
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, cfg=cfg, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img

@MODELS.register_module()
class INRGAN(BaseModule):
    # TODO: add docstring and functions.
    pass
