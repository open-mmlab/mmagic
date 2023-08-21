# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

from mmagic.registry import MODELS


class Downsample2D(nn.Module):
    """A downsampling layer with an optional convolution.

    Args:
        channels (int): channels in the inputs and outputs.
        use_conv (bool): a bool determining if a convolution is applied.
        out_channels (int): output channels
        padding (int): padding num
    """

    def __init__(self,
                 channels,
                 use_conv=False,
                 out_channels=None,
                 padding=1,
                 name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states):
        """forward hidden states."""
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode='constant', value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class Upsample2D(nn.Module):
    """An upsampling layer with an optional convolution.

    Args:
        channels (int): channels in the inputs and outputs.
        use_conv (bool): a bool determining if a convolution is applied.
        use_conv_transpose (bool): whether to use conv transpose.
        out_channels (int): output channels.
    """

    def __init__(self,
                 channels,
                 use_conv=False,
                 use_conv_transpose=False,
                 out_channels=None,
                 name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        else:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)

        self.conv = conv

    def forward(self, hidden_states, output_size=None):
        """forward with hidden states."""
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=2.0, mode='nearest')
        else:
            hidden_states = F.interpolate(
                hidden_states, size=output_size, mode='nearest')

        # TODO(Suraj, Patrick)
        #  - clean up after weight dicts are correctly renamed
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(nn.Module):
    """resnet block support down sample and up sample.

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        conv_shortcut (bool): whether to use conv shortcut.
        dropout (float): dropout rate.
        temb_channels (int): time embedding channels.
        groups (int): conv groups.
        groups_out (int): conv out groups.
        pre_norm (bool): whether to norm before conv. Todo: remove.
        eps (float): eps for groupnorm.
        non_linearity (str): non linearity type.
        time_embedding_norm (str): time embedding norm type.
        output_scale_factor (float): factor to scale input and output.
        use_in_shortcut (bool): whether to use conv in shortcut.
        up (bool): whether to upsample.
        down (bool): whether to downsample.
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity='silu',
        time_embedding_norm='default',
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == 'silu' and \
                digit_version(TORCH_VERSION) > digit_version('1.6.0'):
            self.nonlinearity = nn.SiLU()
        else:
            mmengine.print_log('\'SiLU\' is not supported for '
                               f'torch < 1.6.0, found \'{torch.version}\'.'
                               'Use ReLu instead but result maybe wrong')
            self.nonlinearity = nn.ReLU()

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = \
                Downsample2D(
                    in_channels, use_conv=False, padding=1, name='op')

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut  # noqa

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        """forward with hidden states and time embeddings."""
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes.
            # see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = \
                self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = \
            (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each
    other. Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/
    1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Args:
        channels (int): The number of channels in the input and output.
        num_head_channels (int, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        norm_num_groups (int, *optional*, defaults to 32):
            The number of groups to use for group norm.
        rescale_output_factor (float, *optional*, defaults to 1.0):
            The factor to rescale the output by.
        eps (float, *optional*, defaults to 1e-5):
            The epsilon value to use for group norm.
    """

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1  # noqa
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(
            num_channels=channels,
            num_groups=norm_num_groups,
            eps=eps,
            affine=True)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels, 1)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        """transpose projection."""
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D)
        #  -> (B, T, H, D) -> (B, H, T, D)
        new_projection = \
            projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states):
        """forward hidden states."""
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel,
                                           height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        # get scores
        if self.num_heads > 1:
            query_states = self.transpose_for_scores(query_proj)
            key_states = self.transpose_for_scores(key_proj)
            value_states = self.transpose_for_scores(value_proj)

            attention_scores = torch.matmul(
                query_states, key_states.transpose(-1, -2)) * scale
        else:
            query_states, key_states, value_states = \
                query_proj, key_proj, value_proj

            attention_scores = torch.baddbmm(
                torch.empty(
                    query_states.shape[0],
                    query_states.shape[1],
                    key_states.shape[1],
                    dtype=query_states.dtype,
                    device=query_states.device,
                ),
                query_states,
                key_states.transpose(-1, -2),
                beta=0,
                alpha=scale,
            )

        attention_probs = torch.softmax(
            attention_scores.float(), dim=-1).type(attention_scores.dtype)

        # compute attention output
        if self.num_heads > 1:
            hidden_states = torch.matmul(attention_probs, value_states)
            hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
            new_hidden_states_shape = \
                hidden_states.size()[:-2] + (self.channels,)
            hidden_states = hidden_states.view(new_hidden_states_shape)
        else:
            hidden_states = torch.bmm(attention_probs, value_states)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch, channel, height, width)

        # res connect and rescale
        hidden_states = \
            (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class UNetMidBlock2D(nn.Module):
    """middle block in unet.

    Args:
        in_channels (int): input channels.
        temb_channels (int): time embedding channels.
        dropout (float): dropout rate, defaults to 0.0.
        num_layers (int): layer num.
        resnet_eps (float): resnet eps, defaults to 1e-6.
        resnet_time_scale_shift (str):
            time scale shift, defaults to 'default'.
        resnet_act_fn (str):
            act function in resnet, defaults to 'silu'.
        resnet_groups (int):
            conv groups in resnet, defaults to 32.
        resnet_pre_norm (bool):
            pre norm in resnet, defaults to True.
        attn_num_head_channels (int):
            attention head channels, defaults to 1.
        attention_type (str):
            attention type ,defaults to 'default'.
        output_scale_factor (float):
            output scale factor, defaults to 1.0.
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'silu',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type='default',
        output_scale_factor=1.0,
    ):
        super().__init__()

        self.attention_type = attention_type
        resnet_groups = resnet_groups if resnet_groups is not None else min(
            in_channels // 4, 32)  # noqa

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                AttentionBlock(
                    in_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                ))
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_states=None):
        """forward with hidden states, time embedding and encoder states."""
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if self.attention_type == 'default':
                hidden_states = attn(hidden_states)
            else:
                hidden_states = attn(hidden_states, encoder_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DownEncoderBlock2D(nn.Module):
    """Down encoder block in vae.

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        dropout (float): dropout rate, defaults to 0.0.
        num_layers (int): layer nums, defaults to 1.
        resnet_eps (float): resnet eps, defaults to 1e-6.
        resnet_time_scale_shift (str):
            time scale shift in resnet, defaults to 'default'.
        resnet_act_fn (str):
            act function in resnet, defaults to 'silu'.
        resnet_groups (int):
            group num in resnet, defaults to 32.
        resnet_pre_norm (bool):
            whether to pre norm in resnet, defaults to True.
        output_scale_factor (float):
            output scale factor, defaults to 1.0.
        add_downsample (bool):
            whether to add downsample, defaults to True,
        downsample_padding (int):
            downsample padding num, defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'silu',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    name='op')
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        """forward with hidden states."""
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    """construct encoder in vae."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=('DownEncoderBlock2D', ),
        block_out_channels=(64, ),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn='silu',
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1)

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=0,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift='default',
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6)
        if digit_version(TORCH_VERSION) > digit_version('1.6.0'):
            self.conv_act = nn.SiLU()
        else:
            mmengine.print_log('\'SiLU\' is not supported for '
                               f'torch < 1.6.0, found \'{torch.version}\'.'
                               'Use ReLu instead but result maybe wrong')
            self.conv_act = nn.ReLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        """encoder forward."""
        sample = x
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class UpDecoderBlock2D(nn.Module):
    """construct up decoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(
                    out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        """forward hidden states."""
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    """construct decoder in vae."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=('UpDecoderBlock2D', ),
        block_out_channels=(64, ),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn='silu',
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift='default',
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6)
        if digit_version(TORCH_VERSION) > digit_version('1.6.0'):
            self.conv_act = nn.SiLU()
        else:
            mmengine.print_log('\'SiLU\' is not supported for '
                               f'torch < 1.6.0, found \'{torch.version}\'.'
                               'Use ReLu instead but result maybe wrong')
            self.conv_act = nn.ReLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        """decoder forward."""
        sample = z
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DiagonalGaussianDistribution(object):
    """Calculate diagonal gaussian distribution."""

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype)

    def sample(self, generator: Optional[torch.Generator] = None) \
            -> torch.FloatTensor:
        """sample function."""
        device = self.parameters.device
        sample_device = device
        sample = torch.randn(
            self.mean.shape, generator=generator, device=sample_device)
        # make sure sample is on the same device
        # as the parameters and has same dtype
        sample = sample.to(device=device, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        """calculate kl divergence."""
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var +
                    self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        """calculate negative log likelihood."""
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar +
            torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)  # noqa

    def mode(self):
        """return self.mean."""
        return self.mean


@MODELS.register_module('EditAutoencoderKL')
class AutoencoderKL(nn.Module):
    r"""Variational Autoencoder (VAE) model with KL loss
    from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.

    Args:
        in_channels (int, *optional*, defaults to 3):
            Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3):
            Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
        latent_channels (`int`, *optional*, defaults to `4`):
            Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`):
            sample size is now not supported.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ('DownEncoderBlock2D', ),
        up_block_types: Tuple[str] = ('UpDecoderBlock2D', ),
        block_out_channels: Tuple[int] = (64, ),
        layers_per_block: int = 1,
        act_fn: str = 'silu',
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
    ):
        super().__init__()

        self.block_out_channels = block_out_channels
        self.latent_channels = latent_channels

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = torch.nn.Conv2d(2 * latent_channels,
                                          2 * latent_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_channels,
                                               latent_channels, 1)

    @property
    def dtype(self):
        """The data type of the parameters of VAE."""
        return next(self.parameters()).dtype

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> Dict:
        """encode input."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior, )

        return Dict(latent_dist=posterior)

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) \
            -> Union[Dict, torch.FloatTensor]:
        """decode z."""
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec, )

        return Dict(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[Dict, torch.FloatTensor]:
        """
        Args:
            sample (torch.FloatTensor): Input sample.
            sample_posterior (bool):
                Whether to sample from the posterior.
                defaults to `False`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`Dict`] instead of a plain tuple.
        Returns:
            Dict(sample=dec): decode results.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec, )

        return Dict(sample=dec)
