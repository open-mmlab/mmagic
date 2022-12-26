# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from .attention import Transformer2DModel
from .res_blocks import Downsample2D, ResnetBlock2D, Upsample2D


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_eps=1e-5,
    resnet_groups=32,
    cross_attention_dim=1280,
    downsample_padding=1,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
):
    """get unet down path block."""

    down_block_type = down_block_type[7:] if down_block_type.startswith(
        'UNetRes') else down_block_type
    if down_block_type == 'DownBlock2D':
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
        )
    elif down_block_type == 'CrossAttnDownBlock2D':
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
        )
    raise ValueError(f'{down_block_type} does not exist.')


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_eps=1e-5,
    resnet_groups=32,
    cross_attention_dim=1280,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
):
    """get unet up path block."""

    up_block_type = up_block_type[7:] if up_block_type.startswith(
        'UNetRes') else up_block_type
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
        )
    elif up_block_type == 'CrossAttnUpBlock2D':
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
        )
    raise ValueError(f'{up_block_type} does not exist.')


class UNetMidBlock2DCrossAttn(nn.Module):
    """unet mid block built by cross attention."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type='default',
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
    ):
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(
            in_channels // 4, 32)

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
                Transformer2DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
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

    def set_attention_slice(self, slice_size):
        """set attention slice."""

        head_dims = self.attn_num_head_channels
        head_dims = [head_dims] if isinstance(head_dims, int) else head_dims
        if slice_size is not None and any(dim % slice_size != 0
                                          for dim in head_dims):
            raise ValueError(
                f'Make sure slice_size {slice_size} is a common divisor of '
                f'the number of heads used in cross_attention: {head_dims}')

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        """forward with hidden states."""

        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states).sample
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnDownBlock2D(nn.Module):
    """Down block built by cross attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type='default',
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                ))

        self.attentions = nn.ModuleList(attentions)
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

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        """set attention slice."""

        head_dims = self.attn_num_head_channels
        head_dims = [head_dims] if isinstance(head_dims, int) else head_dims
        if slice_size is not None and any(dim % slice_size != 0
                                          for dim in head_dims):
            raise ValueError(
                f'Make sure slice_size {slice_size} is a common divisor of '
                f'the number of heads used in cross_attention: {head_dims}')

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        """forward with hidden states."""

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states).sample

            output_states += (hidden_states, )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states, )

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """Down block built by resnet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
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
                    temb_channels=temb_channels,
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

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        """forward with hidden states."""

        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states, )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states, )

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    """Up block built by cross attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type='default',
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers -
                                                1) else out_channels
            resnet_in_channels = \
                prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                ))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(
                    out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        """set attention slice."""

        head_dims = self.attn_num_head_channels
        head_dims = [head_dims] if isinstance(head_dims, int) else head_dims
        if slice_size is not None and any(dim % slice_size != 0
                                          for dim in head_dims):
            raise ValueError(
                f'Make sure slice_size {slice_size} is a common divisor of '
                f'the number of heads used in cross_attention: {head_dims}')

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        """forward with hidden states and res hidden states."""

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states],
                                      dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2D(nn.Module):
    """Up block built by resnet."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
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
            res_skip_channels = in_channels if (i == num_layers -
                                                1) else out_channels
            resnet_in_channels = \
                prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
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

        self.gradient_checkpointing = False

    def forward(self,
                hidden_states,
                res_hidden_states_tuple,
                temb=None,
                upsample_size=None):
        """forward with hidden states and res hidden states."""

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states],
                                      dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
