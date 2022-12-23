# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """This matches the implementation in Denoising Diffusion Probabilistic
    Models: Create sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        embedding_dim (int): the dimension of the output.
        flip_sin_to_cos (bool):
            whether to flip sin to cos, defaults to False.
        downscale_freq_shift (float):
            downscale frequecy shift, defaults to 1.
        scale (float):
            embedding scale, defaults to 1.
        max_period: controls the minimum frequency of the exponent.

    Returns:
        emb (torch.Tensor): an [N x dim] Tensor of positional embeddings.
    """

    assert len(timesteps.shape) == 1, 'Timesteps should be a 1d-array'

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    """Module which uses linear to embed timestep."""

    def __init__(self,
                 in_channels: int,
                 time_embed_dim: int,
                 act_fn: str = 'silu',
                 out_dim: int = None):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = None
        if act_fn == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

    def forward(self, sample):
        """forward with sample."""

        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    """A module which transforms timesteps to embedding."""

    def __init__(self,
                 num_channels: int,
                 flip_sin_to_cos: bool = True,
                 downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        """forward with timesteps."""

        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb
