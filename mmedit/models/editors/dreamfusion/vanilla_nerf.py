# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmedit.models.utils import normalize_vecs
from mmedit.registry import MODULES
from mmengine.model import BaseModule

from .activate import trunc_exp

# from .utils import auto_batchicy


class FreqEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 max_freq_log2,
                 N_freqs,
                 log_sampling=True,
                 include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2**torch.linspace(0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2**0, 2**max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, *args, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out


class ResBlock(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU(inplace=True)

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out


class MLP(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 dim_hidden,
                 num_layers,
                 block,
                 bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for idx in range(num_layers):
            if idx == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif idx != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.Sequential(*net)

    def forward(self, x):

        for idx in range(self.num_layers):
            x = self.net[idx](x)

        return x


class VanillaMLP(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 use_res_block: bool = False,
                 use_bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        net = [BasicBlock(in_channels, hidden_channels, bias=use_bias)]
        block = ResBlock if use_res_block else BasicBlock
        for _ in range(num_layers - 2):
            net.append(block(hidden_channels, hidden_channels, bias=use_bias))
        net.append(nn.Linear(hidden_channels, out_channels, bias=use_bias))

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODULES.register_module('VanillaNeRF')
class NeRFNetwork(BaseModule):

    # TODO: optim n_freq and max_freq_log2

    def __init__(
            self,
            num_layers: int = 4,  # 5 in paper
            hidden_dim: int = 96,  # 128 in paper
            bg_radius: float = 1.4,
            num_layers_bg: int = 2,  # 3 in paper
            hidden_dim_bg: int = 64,  # 64 in paper
            n_freq: int = 6,
            max_freq_log2: int = 5,
            bg_freq: int = 4,
            bg_max_freq_log2: int = 3,
            init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.bound = 1
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder = FreqEncoder(
            input_dim=3, max_freq_log2=max_freq_log2, N_freqs=n_freq)
        self.in_dim = self.encoder.output_dim
        self.sigma_net = VanillaMLP(
            self.in_dim, 4, hidden_dim, num_layers, use_res_block=True)

        # background network
        self.bg_radius = bg_radius
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            # multires = 4
            self.encoder_bg = FreqEncoder(
                input_dim=3, max_freq_log2=bg_max_freq_log2, N_freqs=bg_freq)
            self.in_dim_bg = self.encoder_bg.output_dim
            self.bg_net = VanillaMLP(self.in_dim_bg, 3, hidden_dim_bg,
                                     num_layers_bg)
        else:
            self.encoder_bg = self.bg_net = None

    def spatial_density_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial density bias in Equal (9) in appendix."""

        # Spatial density bias.
        d = (x**2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2**2))

        return g

    def forward_bg(self, d: torch.Tensor) -> torch.Tensor:
        """Forward functionfor the background network."""
        if self.bg_radius == 0:
            return torch.rand_like(d)

        h = self.encoder_bg(d)  # [N, C]

        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def forward_fg(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the foreground network."""
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = self.encoder(x, bound=self.bound)
        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0] + self.spatial_density_bias(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the normal with density."""

        with torch.enable_grad():
            x.requires_grad_(True)
            sigma, _ = self.forward_fg(x)
            normal = -torch.autograd.grad(
                torch.sum(sigma), x, create_graph=True)[0]  # [N, 3]

        normal = normalize_vecs(normal, clamp_eps=1e-20)

        return normal

    # @auto_batchicy(no_batchify_args='light_d')
    def forward(self,
                x: torch.Tensor,
                light_d: Optional[torch.Tensor] = None,
                ambient_ratio: float = 1,
                shading: str = 'albedo') -> dict:
        """The forward function."""
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only),
        #   0 == only shading (textureless)

        # NOTE: a dirty way to only get albedo in visualization step
        if shading == 'albedo' or not self.training:
            sigma, albedo = self.forward_fg(x)
            output = dict(sigma=sigma, color=albedo)
        else:
            with torch.enable_grad():
                x.requires_grad_(True)
                sigma, albedo = self.forward_fg(x)
                # query gradient
                normal = -torch.autograd.grad(
                    torch.sum(sigma), x, create_graph=True)[0]  # [N, 3]

            normal = normalize_vecs(normal, clamp_eps=1e-20)

            if shading == 'normal':
                color = (normal + 1) / 2
            else:
                # lambertian shading
                lambertian = ambient_ratio + (1 - ambient_ratio) * (
                    normal @ light_d).clamp(min=0)  # [N,]
                if shading == 'textureless':
                    color = lambertian.unsqueeze(-1).repeat(1, 3)
                else:  # 'lambertian'
                    color = albedo * lambertian.unsqueeze(-1)
            output = dict(sigma=sigma, color=color, normal=normal)

        return output

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        sigma, albedo = self.forward_fg(x)
        return dict(sigma=sigma, color=albedo)
