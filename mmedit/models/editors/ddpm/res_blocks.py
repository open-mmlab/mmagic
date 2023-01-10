# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version


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
            self.downsample = Downsample2D(
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
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None,
                                                               None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor +
                         hidden_states) / self.output_scale_factor

        return output_tensor


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

        hidden_states = self.conv(hidden_states)

        return hidden_states


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
        """forward with hidden states."""
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode='constant', value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states
