# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from functools import partial

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import build_norm_layer
from mmengine.model import constant_init
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

from mmedit.registry import MODELS, MODULES


class EmbedSequential(nn.Sequential):
    """A sequential module that passes timestep embeddings to the children that
    support it as an extra input.

    Modified from
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py#L35
    """

    def forward(self, x, y):
        for layer in self:
            if isinstance(layer, DenoisingResBlock):
                x = layer(x, y)
            else:
                x = layer(x)
        return x


@MODELS.register_module('GN32')
class GroupNorm32(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=32, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def convert_module_to_f16(layer):
    """Convert primitive modules to float16."""
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        layer.weight.data = layer.weight.data.half()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.half()


def convert_module_to_f32(layer):
    """Convert primitive modules to float32, undoing
    convert_module_to_f16()."""
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        layer.weight.data = layer.weight.data.float()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.float()


@MODELS.register_module()
class SiLU(nn.Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.
    Args:
        input (bool, optional): Use inplace operation or not.
            Defaults to `False`.
    """

    def __init__(self, inplace=False):
        super().__init__()
        if digit_version(TORCH_VERSION) <= digit_version('1.6.0') and inplace:
            mmengine.print_log(
                'Inplace version of \'SiLU\' is not supported for '
                f'torch < 1.6.0, found \'{torch.version}\'.')
        self.inplace = inplace

    def forward(self, x):
        """Forward function for SiLU.
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after activation.
        """

        if digit_version(TORCH_VERSION) <= digit_version('1.6.0'):
            return x * torch.sigmoid(x)

        return F.silu(x, inplace=self.inplace)



@MODULES.register_module()
class MultiHeadAttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each
    other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self,
                 in_channels,
                 num_heads=1,
                 num_head_channels=-1,
                 use_new_attention_order=False,
                 norm_cfg=dict(type='GN32', num_groups=32)):
        super().__init__()
        self.in_channels = in_channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (in_channels % num_head_channels == 0), (
                f'q,k,v channels {in_channels} is not divisible by '
                'num_head_channels {num_head_channels}')
            self.num_heads = in_channels // num_head_channels
        _, self.norm = build_norm_layer(norm_cfg, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)

        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


@MODULES.register_module()
class QKVAttentionLegacy(nn.Module):
    """A module which performs QKV attention.

    Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(
            ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            'bct,bcs->bts', q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)


@MODULES.register_module()
class QKVAttention(nn.Module):
    """A module which performs QKV attention and splits in a different
    order."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            'bct,bcs->bts',
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight,
                         v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


@MODULES.register_module()
class TimeEmbedding(nn.Module):
    """Time embedding layer, reference to Two level embedding. First embedding
    time by an embedding function, then feed to neural networks.

    Args:
        in_channels (int): The channel number of the input feature map.
        embedding_channels (int): The channel number of the output embedding.
        embedding_mode (str, optional): Embedding mode for the time embedding.
            Defaults to 'sin'.
        embedding_cfg (dict, optional): Config for time embedding.
            Defaults to None.
        act_cfg (dict, optional): Config for activation layer. Defaults to
            ``dict(type='SiLU', inplace=False)``.
    """

    def __init__(self,
                 in_channels,
                 embedding_channels,
                 embedding_mode='sin',
                 embedding_cfg=None,
                 act_cfg=dict(type='SiLU', inplace=False)):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(in_channels, embedding_channels), MODELS.build(act_cfg),
            nn.Linear(embedding_channels, embedding_channels))

        # add `dim` to embedding config
        embedding_cfg_ = dict(dim=in_channels)
        if embedding_cfg is not None:
            embedding_cfg_.update(embedding_cfg)
        if embedding_mode.upper() == 'SIN':
            self.embedding_fn = partial(self.sinusodial_embedding,
                                        **embedding_cfg_)
        else:
            raise ValueError('Only support `SIN` for time embedding, '
                             f'but receive {embedding_mode}.')

    @staticmethod
    def sinusodial_embedding(timesteps, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps (torch.Tensor): Timestep to embedding. 1-D tensor shape
                as ``[bz, ]``,  one per batch element.
            dim (int): The dimension of the embedding.
            max_period (int, optional): Controls the minimum frequency of the
                embeddings. Defaults to ``10000``.

        Returns:
            torch.Tensor: Embedding results shape as `[bz, dim]`.
        """

        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """Forward function for time embedding layer.
        Args:
            t (torch.Tensor): Input timesteps.

        Returns:
            torch.Tensor: Timesteps embedding.

        """
        return self.blocks(self.embedding_fn(t))


@MODULES.register_module()
class DenoisingResBlock(nn.Module):
    """Resblock for the denoising network. If `in_channels` not equals to
    `out_channels`, a learnable shortcut with conv layers will be added.

    Args:
        in_channels (int): Number of channels of the input feature map.
        embedding_channels (int): Number of channels of the input embedding.
        use_scale_shift_norm (bool): Whether use scale-shift-norm in
            `NormWithEmbedding` layer.
        dropout (float): Probability of the dropout layers.
        out_channels (int, optional): Number of output channels of the
            ResBlock. If not defined, the output channels will equal to the
            `in_channels`. Defaults to `None`.
        norm_cfg (dict, optional): The config for the normalization layers.
            Defaults too ``dict(type='GN', num_groups=32)``.
        act_cfg (dict, optional): The config for the activation layers.
            Defaults to ``dict(type='SiLU', inplace=False)``.
        shortcut_kernel_size (int, optional): The kernel size for the shortcut
            conv. Defaults to ``1``.
    """

    def __init__(self,
                 in_channels,
                 embedding_channels,
                 use_scale_shift_norm,
                 dropout,
                 out_channels=None,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1,
                 up=False,
                 down=False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        _norm_cfg = deepcopy(norm_cfg)

        _, norm_1 = build_norm_layer(_norm_cfg, in_channels)
        conv_1 = [
            norm_1,
            MODELS.build(act_cfg),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        ]
        self.conv_1 = nn.Sequential(*conv_1)

        norm_with_embedding_cfg = dict(
            in_channels=out_channels,
            embedding_channels=embedding_channels,
            use_scale_shift=use_scale_shift_norm,
            norm_cfg=_norm_cfg)
        self.norm_with_embedding = MODULES.build(
            dict(type='NormWithEmbedding'),
            default_args=norm_with_embedding_cfg)

        conv_2 = [
            MODELS.build(act_cfg),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        ]
        self.conv_2 = nn.Sequential(*conv_2)

        assert shortcut_kernel_size in [
            1, 3
        ], ('Only support `1` and `3` for `shortcut_kernel_size`, but '
            f'receive {shortcut_kernel_size}.')

        self.learnable_shortcut = out_channels != in_channels

        if self.learnable_shortcut:
            shortcut_padding = 1 if shortcut_kernel_size == 3 else 0
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                shortcut_kernel_size,
                padding=shortcut_padding)

        self.updown = up or down

        if up:
            self.h_upd = DenoisingUpsample(in_channels, False)
            self.x_upd = DenoisingUpsample(in_channels, False)
        elif down:
            self.h_upd = DenoisingDownsample(in_channels, False)
            self.x_upd = DenoisingDownsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.init_weights()

    def forward_shortcut(self, x):
        if self.learnable_shortcut:
            return self.shortcut(x)
        return x

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.
            y (torch.Tensor): Shared time embedding or shared label embedding.

        Returns:
            torch.Tensor : Output feature map tensor.
        """
        if self.updown:
            in_rest, in_conv = self.conv_1[:-1], self.conv_1[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.conv_1(x)

        shortcut = self.forward_shortcut(x)
        h = self.norm_with_embedding(h, y)
        h = self.conv_2(h)
        return h + shortcut

    def init_weights(self):
        # apply zero init to last conv layer
        constant_init(self.conv_2[-1], 0)


@MODULES.register_module()
class NormWithEmbedding(nn.Module):
    """Nornalization with embedding layer. If `use_scale_shift == True`,
    embedding results will be chunked and used to re-shift and re-scale
    normalization results. Otherwise, embedding results will directly add to
    input of normalization layer.

    Args:
        in_channels (int): Number of channels of the input feature map.
        embedding_channels (int) Number of channels of the input embedding.
        norm_cfg (dict, optional): Config for the normalization operation.
            Defaults to `dict(type='GN', num_groups=32)`.
        act_cfg (dict, optional): Config for the activation layer. Defaults
            to `dict(type='SiLU', inplace=False)`.
        use_scale_shift (bool): If True, the output of Embedding layer will be
            split to 'scale' and 'shift' and map the output of normalization
            layer to ``out * (1 + scale) + shift``. Otherwise, the output of
            Embedding layer will be added with the input before normalization
            operation. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 embedding_channels,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 use_scale_shift=True):
        super().__init__()
        self.use_scale_shift = use_scale_shift
        _, self.norm = build_norm_layer(norm_cfg, in_channels)

        embedding_output = in_channels * 2 if use_scale_shift else in_channels
        self.embedding_layer = nn.Sequential(
            MODELS.build(act_cfg),
            nn.Linear(embedding_channels, embedding_output))

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.
            y (torch.Tensor): Shared time embedding or shared label embedding.

        Returns:
            torch.Tensor : Output feature map tensor.
        """
        embedding = self.embedding_layer(y).type(x.dtype)
        embedding = embedding[:, :, None, None]
        if self.use_scale_shift:
            scale, shift = torch.chunk(embedding, 2, dim=1)
            x = self.norm(x)
            x = x * (1 + scale) + shift
        else:
            x = self.norm(x + embedding)
        return x


@MODULES.register_module()
class DenoisingDownsample(nn.Module):
    """Downsampling operation used in the denoising network. Support average
    pooling and convolution for downsample operation.

    Args:
        in_channels (int): Number of channels of the input feature map to be
            downsampled.
        with_conv (bool, optional): Whether use convolution operation for
            downsampling.  Defaults to `True`.
    """

    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        if with_conv:
            self.downsample = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Forward function for downsampling operation.
        Args:
            x (torch.Tensor): Feature map to downsample.

        Returns:
            torch.Tensor: Feature map after downsampling.
        """
        return self.downsample(x)


@MODULES.register_module()
class DenoisingUpsample(nn.Module):
    """Upsampling operation used in the denoising network. Allows users to
    apply an additional convolution layer after the nearest interpolation
    operation.

    Args:
        in_channels (int): Number of channels of the input feature map to be
            downsampled.
        with_conv (bool, optional): Whether apply an additional convolution
            layer after upsampling.  Defaults to `True`.
    """

    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        """Forward function for upsampling operation.
        Args:
            x (torch.Tensor): Feature map to upsample.

        Returns:
            torch.Tensor: Feature map after upsampling.
        """
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x
