# Copyright (c) OpenMMLab. All rights reserved.
import math

import mmengine
import torch
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version
from torch import nn


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
        if act_fn == 'silu' and \
                digit_version(TORCH_VERSION) > digit_version('1.6.0'):
            self.act = nn.SiLU()
        else:
            mmengine.print_log('\'SiLU\' is not supported for '
                               f'torch < 1.6.0, found \'{torch.version}\'.'
                               'Use ReLu instead but result maybe wrong')
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
        self.max_period = 10000
        self.scale = 1

    def forward(self, timesteps):
        """forward with timesteps."""

        assert len(timesteps.shape) == 1, 'Timesteps should be a 1d-array'

        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * \
            torch.arange(
                start=0,
                end=half_dim,
                dtype=torch.float32,
                device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # scale embeddings
        emb = self.scale * emb

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # flip sine and cosine embeddings
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        # zero pad
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
