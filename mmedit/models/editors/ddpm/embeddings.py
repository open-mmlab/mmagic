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
        elif act_fn == 'mish':
            self.act = nn.Mish()

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


class ImagePositionalEmbeddings(nn.Module):
    """Converts latent image classes into vector embeddings. Sums the vector
    embeddings with positional embeddings for the height and width of the
    latent space.

    For more details, see figure 10 of the dall-e paper:
    https://arxiv.org/abs/2102.12092

    For VQ-diffusion:

    Output vector embeddings are used as input for the transformer.

    Note that the vector embeddings for the transformer are
    different than the vector embeddings from the VQVAE.

    Args:
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        embed_dim (`int`):
            Dimension of the produced vector embeddings.
            Used for the latent pixel, height, and width embeddings.
    """

    def __init__(
        self,
        num_embed: int,
        height: int,
        width: int,
        embed_dim: int,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        self.height_emb = nn.Embedding(self.height, embed_dim)
        self.width_emb = nn.Embedding(self.width, embed_dim)

    def forward(self, index):
        """forward with index."""

        emb = self.emb(index)

        height_emb = self.height_emb(
            torch.arange(self.height,
                         device=index.device).view(1, self.height))

        # 1 x H x D -> 1 x H x 1 x D
        height_emb = height_emb.unsqueeze(2)

        width_emb = self.width_emb(
            torch.arange(self.width, device=index.device).view(1, self.width))

        # 1 x W x D -> 1 x 1 x W x D
        width_emb = width_emb.unsqueeze(1)

        pos_emb = height_emb + width_emb

        # 1 x H x W x D -> 1 x L xD
        pos_emb = pos_emb.view(1, self.height * self.width, -1)

        emb = emb + pos_emb[:, :emb.shape[1], :]

        return emb
