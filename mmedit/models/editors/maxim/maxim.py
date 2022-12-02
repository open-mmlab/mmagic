# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from mmengine.model import BaseModel

from mmedit.registry import MODELS
from .maxim_modules import Conv1x1, Conv3x3, CrossGatingBlock


@MODELS.register_module()
class MAXIM(BaseModel):
    """The MAXIM model function with multi-stage and multi-scale supervision.

    For more model details, please check the CVPR paper:
    MAXIM: MUlti-Axis MLP for Image Processing
    (https://arxiv.org/abs/2201.02973)

    This is a pytorch reimplementation for MAXIM.

    Args:
        features: initial hidden dimension for the input resolution.
        depth: the number of downsampling depth for the model.
        num_stages: how many stages to use. Also affects the output list.
        num_groups: how many blocks each stage contains.
        num_bottleneck_blocks: how many bottleneck blocks.
        block_gmlp_factor: the input projection factor for block_gMLP layers.
        grid_gmlp_factor: the input projection factor for grid_gMLP layers.
        input_proj_factor: the input projection factor for the MAB block.
        channels_reduction: the channel reduction factor for SE layer.
        use_bias: whether to use bias in all the conv/mlp layers.
        num_supervision_scales: the number of desired supervision scales.
        lrelu_slope: the negative slope parameter in leaky_relu layers.
        use_global_mlp: whether to use the multi-axis gated
        MLP block (MAB) in each layer.
        use_cross_gating: whether to use the cross-gating
        MLP block (CGB) in the
        skip connections and multi-stage feature fusion layers.
        high_res_stages: how many stages are specified as high-res stages.
        The rest (depth - high_res_stages) are called low_res_stages.
        block_size_hr: the block_size parameter for high-res stages.
        block_size_lr: the block_size parameter for low-res stages.
        grid_size_hr: the grid_size parameter for high-res stages.
        grid_size_lr: the grid_size parameter for low-res stages.
        num_outputs: the output channels.
        dropout_rate: Dropout rate.

    Returns:
    The output contains a list of arrays consisting of multi-stage multi-scale
    outputs. For example, if num_stages = num_supervision_scales = 3 (the
    model used in the paper), the output specs are: outputs =
    [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
     [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
     [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
    The final output can be retrieved by outputs[-1][-1].
    """

    def __init__(self,
                 features: int = 64,
                 depth: int = 3,
                 num_stages: int = 2,
                 num_groups: int = 1,
                 num_bottleneck_blocks: int = 1,
                 block_gmlp_factor: int = 2,
                 grid_gmlp_factor: int = 2,
                 input_proj_factor: int = 2,
                 channels_reduction: int = 4,
                 use_bias: bool = True,
                 num_supervision_scales: int = 1,
                 lrelu_slope: float = 0.2,
                 use_global_mlp: bool = True,
                 use_cross_gating: bool = True,
                 high_res_stages: int = 2,
                 block_size_hr: Sequence[int] = (16, 16),
                 block_size_lr: Sequence[int] = (8, 8),
                 grid_size_hr: Sequence[int] = (16, 16),
                 grid_size_lr: Sequence[int] = (8, 8),
                 num_inputs: int = 3,
                 num_outputs: int = 3,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.features = features
        self.depth = depth
        self.num_stages = num_stages
        self.num_groups = num_groups
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.use_bias = use_bias
        self.num_supervision_scales = num_supervision_scales
        self.lrelu_slope = lrelu_slope
        self.use_global_mlp = use_global_mlp
        self.use_cross_gating = use_cross_gating
        self.high_res_stages = high_res_stages
        self.block_size_hr = block_size_hr
        self.block_size_lr = block_size_lr
        self.grid_size_hr = grid_size_hr
        self.grid_size_lr = grid_size_lr
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate
        self._model_setup()

    def _model_setup(self):
        # Input convolution, get multi-scale input feature
        self.conv1 = nn.ModuleList()
        self.gating_block = nn.ModuleList()
        for i in range(self.num_supervision_scales):
            self.conv1.append(
                Conv3x3(
                    self.num_inputs, (2**i) * self.features,
                    bias=self.use_bias,
                    padding=1))

            if self.use_cross_gating:
                block_size = self.block_size_hr \
                    if i < self.high_res_stages else self.block_size_lr
                grid_size = self.grid_size_hr \
                    if i < self.high_res_stages else self.block_size_lr
                self.gating_block.append(
                    CrossGatingBlock(
                        features=(2**i) * self.features,
                        block_size=block_size,
                        grid_size=grid_size,
                        dropout_rate=self.dropout_rate,
                        input_proj_factor=self.input_proj_factor,
                        upsample_y=False,
                        use_bias=self.use_bias,
                    ))
            else:
                self.gating_block.append(
                    Conv1x1((2**i) * self.features, use_bias=self.use_bias))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape  # input image shape
        shortcuts = []
        shortcuts.append(x)
        # Get multi-scale input images
        for i in range(1, self.num_supervision_scales):
            shortcuts.append(
                nn.functional.interpolate(
                    x, size=(B, C, H // (2**i), W // (2**i)), mode='nearest'))

        # store outputs from all stages and all scales
        # Eg, [[(64, 64, 3), (128, 128, 3), (256, 256, 3)],   # Stage-1 outputs
        #      [(64, 64, 3), (128, 128, 3), (256, 256, 3)],]  # Stage-2 outputs
        # outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []
        for idx_stage in range(self.num_stages):
            # Input convolution, get multi-scale input features
            x_scales = []
            for i in range(self.num_supervision_scales):
                x_scale = self.conv1[i](shortcuts[i])
                # If later stages,
                # fuse input features with SAM features from prev stage
                if idx_stage > 0:
                    # use larger blocksize at high-res stages
                    if self.use_cross_gating:
                        x_scale, _ = self.gating_block[i](x_scale,
                                                          sam_features.pop())
                    else:
                        x_scale = self.gating_block[i](
                            torch.cat([x_scale, sam_features.pop()], dim=1))

                x_scales.append(x_scale)

        # not finished, only for commit
        encs_prev, decs_prev = decs_prev, encs_prev

        return x_scales
