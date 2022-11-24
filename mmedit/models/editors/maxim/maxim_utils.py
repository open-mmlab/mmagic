# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Sequence

import torch.nn as nn
from mmengine.model import BaseModel

Conv3x3 = functools.partial(nn.Conv2d, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv2d, kernel_size=(1, 1))
ConvT_up = functools.partial(
    nn.ConvTranspose2d, kernel_size=(2, 2), strides=(2, 2))
Conv_down = functools.partial(nn.Conv, kernel_size=(4, 4), strides=(2, 2))


class CrossGatingBlock(BaseModel):
    """Cross-gating MLP block."""

    def __init__(self,
                 features: int,
                 block_size: Sequence[int],
                 grid_size: Sequence[int],
                 dropout_rate: float = 0.0,
                 input_proj_factor: int = 2,
                 upsample_y: bool = True,
                 use_bias: bool = True):
        super().__init__()
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = self.upsample_y
        self.use_bias = use_bias

        if self.upsample_y:
            self.y_conv1 = ConvT_up(self.features, use_bias=use_bias)
        else:
            self.y_conv1 = ConvT_up(self.features, use_bias=use_bias)
