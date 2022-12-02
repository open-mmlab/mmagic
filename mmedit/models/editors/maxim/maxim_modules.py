# Copyright (c) OpenMMLab. All rights reserved.
import functools
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
from mmengine.model import BaseModel

from .maxim_functions import block_images_einops, unblock_images_einops

Conv3x3 = functools.partial(nn.Conv2d, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv2d, kernel_size=(1, 1))
ConvT_up = functools.partial(
    nn.ConvTranspose2d, kernel_size=(2, 2), strides=(2, 2))
Conv_down = functools.partial(nn.Conv2d, kernel_size=(4, 4), strides=(2, 2))


class CrossGatingBlock(BaseModel):
    """Cross-gating MLP block.

    Args:
        in_dim: list, should be [H, W]
        features: int,
        block_size: Sequence[int],
        grid_size: Sequence[int],
        dropout_rate: float = 0.0,
        input_proj_factor: int = 2,
        upsample_y: bool = True,
        use_bias: bool = True
    """

    def __init__(self,
                 in_dim: list,
                 features: int,
                 block_size: Sequence[int],
                 grid_size: Sequence[int],
                 dropout_rate: float = 0.0,
                 input_proj_factor: int = 2,
                 upsample_y: bool = True,
                 use_bias: bool = True):
        super().__init__()
        self.in_dim = in_dim + [features]
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = self.upsample_y
        self.use_bias = use_bias

        # TODO: input feature != features
        # input shape: [B, C, H, W]
        if self.upsample_y:
            self.y_conv0 = ConvT_up(features, features, bias=use_bias)

        self.x_conv1 = Conv1x1(features, features, bias=use_bias)
        self.y_conv1 = Conv1x1(features, features, bias=use_bias)

        # input shape [B, H, W, C]
        self.x_gating_module = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(features, features * input_proj_factor, bias=use_bias),
            nn.GELU(),
        )
        self.x_getSpatialGatingWeights = GetSpatialGatingWeights(
            features=features,
            block_size=self.block_size,
            grid_size=self.grid_size,
            dropout_rate=self.dropout_rate,
            bias=self.use_bias)
        self.y_gating_module = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(features, features * input_proj_factor, bias=use_bias),
            nn.GELU(),
        )
        self.y_getSpatialGatingWeights = GetSpatialGatingWeights(
            features=features,
            block_size=self.block_size,
            grid_size=self.grid_size,
            dropout_rate=self.dropout_rate,
            bias=self.use_bias)

        # Apply cross gating: X = X * GY, Y = Y * GX
        self.x_last_layer = nn.Linear(features, features, use_bias=use_bias)
        self.x_dropout = nn.Dropout(dropout_rate)

        self.y_last_layer = nn.Linear(features, features, use_bias=use_bias)
        self.y_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, y):
        # Upscale Y signal, y is the gating signal.
        # xy shape: [B, C, H, W]
        if self.upsample_y:
            y = self.y_conv0(y)

        x = self.x_conv1(x)
        y = self.y_conv1(y)
        assert x.shape == y.shape

        # xy shape [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        shortcut_x = deepcopy(x)
        shortcut_y = deepcopy(y)

        # Get gating weights from X
        x = self.x_gating_module(x)
        gx = self.x_getSpatialGatingWeights(x)

        # Get gating weights from Y
        y = self.y_gating_module(y)
        gy = self.y_getSpatialGatingWeights(y)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = self.y_last_layer(y)
        y = self.y_dropout(y)
        y = y + shortcut_y

        x = x * gy  # gating x using y
        x = self.x_last_layer(x)
        x = self.x_dropout(x)
        x = x + y + shortcut_x
        return x, y


class GetSpatialGatingWeights(BaseModel):
    """Get gating weights for cross-gating MLP block.

    Args:
        in_dim: list, should be [H, W]
        features: int
        block_size: Sequence[int]
        grid_size: Sequence[int]
        input_proj_factor: int = 2
        dropout_rate: float = 0.0
        use_bias: bool = True
    """

    def __init__(self,
                 in_dim: list,
                 features: int,
                 block_size: Sequence[int],
                 grid_size: Sequence[int],
                 input_proj_factor: int = 2,
                 dropout_rate: float = 0.0,
                 use_bias: bool = True):
        super().__init__()
        self.in_dim = in_dim + [features]
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.input_proj_factor = input_proj_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        H, W = in_dim
        # input projection
        # input shape: [B, H, W, C]
        self.input_project = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(features, features * input_proj_factor, bias=use_bias),
            nn.GELU())

        # Get grid MLP weights
        self.u_gh, self.u_gw = self.grid_size
        self.u_fh, self.u_fw = H // self.u_gh, W // self.u_gw
        self.grid_layer = nn.Linear(
            self.u_gh * self.u_gw, self.u_gh * self.u_gw, bias=use_bias)

        # Get Block MLP weights
        self.v_fh, self.v_fw = self.block_size
        self.v_gh, self.v_gw = H // self.v_fh, W // self.v_fw
        self.block_layer = nn.Linear(
            self.u_gh * self.u_gw, self.v_fh * self.v_fw, bias=use_bias)

        self.last_layer = nn.Linear(features * 2, features, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, H, W, C = x.shape
        # input projection
        x = self.input_project(x)
        u, v = torch.chunk(x, 2, dim=-1)

        # Get grid MLP weights
        u = block_images_einops(u, patch_size=(self.u_fh, self.u_fw))
        u = torch.swapaxes(u, -1, -3)
        u = self.grid_layer(u)
        u = torch.swapaxes(u, -1, -3)
        u = unblock_images_einops(
            u,
            grid_size=(self.u_gh, self.u_gw),
            patch_size=(self.u_fh, self.u_fw))

        # Get Block MLP weights
        v = block_images_einops(v, patch_size=(self.v_fh, self.v_fw))
        v = torch.swapaxes(v, -1, -2)
        v = self.block_layer(v)
        v = torch.swapaxes(v, -1, -2)
        v = unblock_images_einops(
            v,
            grid_size=(self.v_gh, self.v_gw),
            patch_size=(self.v_fh, self.v_fw))

        x = torch.cat([u, v], dim=-1)
        x = self.last_layer(x)
        x = self.dropout(x)
        return x
