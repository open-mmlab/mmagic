# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from addict import Dict
from torch import nn


class Transformer2DModel(nn.Module):
    """Transformer model for image-like data. Takes either discrete (classes of
    vector embeddings) or continuous (actual embeddings) inputs.

    When input is continuous: First, project the input
     (aka embedding) and reshape to b, t, d. Then apply standard
    transformer action. Finally, reshape to image.

    When input is discrete: First, input (classes of latent pixels)
     is converted to embeddings and has positional
    embeddings applied, see `ImagePositionalEmbeddings`.
    Then apply standard transformer action. Finally, predict
    classes of unnoised image.

    Note that it is assumed one of the input classes is
    the masked latent pixel. The predicted classes of the unnoised
    image do not contain a prediction for the masked pixel as
    the unnoised image cannot be masked.

    Args:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous.
            The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability to use.
        norm_num_groups (int):
            Norm group num, defaults to 32.
        cross_attention_dim (`int`, *optional*):
            The number of context dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain
            a bias parameter.
        sample_size (`int`, *optional*):
            Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for
            learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of
            the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward.
        use_linear_projection (bool):
            Whether to use linear projection, defaults to False.
        only_cross_attention (bool):
            whether only use cross attention, defaults to False.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = 'geglu',
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continuous
        # images of shape `(batch_size, num_channels, width, height)`
        # as well as quantized image embeddings of shape
        # `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete
        # depending on configuration
        self.is_input_continuous = in_channels is not None
        self.is_input_vectorized = num_vector_embeds is not None

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f'Cannot define both `in_channels`: {in_channels} '
                f'and `num_vector_embeds`: {num_vector_embeds}. Make'
                f' sure that either `in_channels` or `num_vector_embeds` '
                'is None.')
        elif not self.is_input_continuous and not self.is_input_vectorized:
            raise ValueError(
                f'Has to define either `in_channels`: {in_channels} or'
                f' `num_vector_embeds`: {num_vector_embeds}. Make'
                f' sure that either `in_channels` or '
                '`num_vector_embeds` is not None.')

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels,
                eps=1e-6,
                affine=True)
            if use_linear_projection:
                self.proj_in = nn.Linear(in_channels, inner_dim)
            else:
                self.proj_in = nn.Conv2d(
                    in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError('input_vectorized not supported now.')

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
            ) for d in range(num_layers)
        ])

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def _set_attention_slice(self, slice_size):
        """set attention slice."""

        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self,
                hidden_states,
                encoder_hidden_states=None,
                timestep=None,
                return_dict: bool = True):
        """forward function.

        Args:
            hidden_states ( When discrete, `torch.LongTensor`
                of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `
                (batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape
                `(batch size, context dim)`, *optional*):
                Conditional embeddings for cross attention layer.
                If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding
                in AdaLayerNorm's. Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`models.unet_2d_condition.UNet2DConditionOutput`]
                instead of a plain tuple.

        Returns:
            Dict if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample
            tensor.
        """
        # 1. Input
        if self.is_input_continuous:
            batch, channel, height, weight = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * weight, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * weight, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        else:
            raise ValueError('input_vectorized not supported now.')

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                context=encoder_hidden_states,
                timestep=timestep)

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight,
                                      inner_dim).permute(0, 3, 1,
                                                         2).contiguous())
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight,
                                      inner_dim).permute(0, 3, 1,
                                                         2).contiguous())

        output = hidden_states + residual

        if not return_dict:
            return (output, )

        return Dict(sample=output)


class BasicTransformerBlock(nn.Module):
    """A basic Transformer block.

    Args:
        dim (int): The number of channels in the input and output.
        num_attention_heads (int): The number of heads to use for
         multi-head attention.
        attention_head_dim (int): The number of channels in each head.
        dropout (float, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (int, *optional*):
            The size of the context vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward.
        attention_bias (bool, *optional*, defaults to `False`):
            Configure if the attentions should contain a bias parameter.
        only_cross_attention (bool, defaults to False):
            whether to use cross attention only.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = 'geglu',
        attention_bias: bool = False,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim
            if only_cross_attention else None,
        )  # is a self-attention
        self.ff = FeedForward(
            dim, dropout=dropout, activation_fn=activation_fn)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )  # is self-attn if context is none

        # layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def _set_attention_slice(self, slice_size):
        """set attention slice."""
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None, timestep=None):
        """forward with hidden states, context and timestep."""
        # 1. Self-Attention
        norm_hidden_states = (self.norm1(hidden_states))

        if self.only_cross_attention:
            hidden_states = self.attn1(norm_hidden_states,
                                       context) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = (self.norm2(hidden_states))
        hidden_states = self.attn2(
            norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Args:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context.
            If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key,
            and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim  # noqa

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        """reshape heads num to batch dim."""
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size,
                                dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size,
                                                    seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        """reshape batch dim to heads num."""
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len,
                                dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size,
                                                    seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        """forward with hidden states, context and mask."""
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember
        # to re-implement when used

        # attention, what we cannot get enough of
        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value,
                                                   sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value):
        """attention calculation."""
        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output

        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        """sliced attention calculation."""
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads),
            device=query.device,
            dtype=query.dtype)
        slice_size = self._slice_size if self._slice_size is not None \
            else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = torch.baddbmm(
                torch.empty(
                    slice_size,
                    query.shape[1],
                    key.shape[1],
                    dtype=query.dtype,
                    device=query.device),
                query[start_idx:end_idx],
                key[start_idx:end_idx].transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Args:
        dim (int): The number of channels in the input.
        dim_out (int, *optional*):
            The number of channels in the output.
            If not given, defaults to `dim`.
        mult (int, *optional*, defaults to 4):
            The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = 'geglu',
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == 'geglu':
            geglu = GEGLU(dim, inner_dim)
        elif activation_fn == 'geglu-approximate':
            geglu = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(geglu)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        """forward with hidden states."""
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function
    from https://arxiv.org/abs/2002.05202.

    Args:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        """gelu activation."""
        return F.gelu(gate)

    def forward(self, hidden_states):
        """forward with hidden states."""
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        """forward function."""
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)
