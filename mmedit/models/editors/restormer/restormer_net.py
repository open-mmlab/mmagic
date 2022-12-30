# Copyright (c) OpenMMLab. All rights reserved.

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmengine.model import BaseModule

from mmedit.registry import MODELS


def to_3d(x):
    """Reshape input tensor."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Reshape input tensor."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(BaseModule):
    """Layer normalization without bias.

    Args:
        normalized_shape (tuple): The shape of inputs.
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(BaseModule):
    """Layer normalization with bias. The bias can be learned.

    Args:
        normalized_shape (tuple): The shape of inputs.
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(BaseModule):
    """Layer normalization module.

    Note: This is different from the layernorm2d in pytorch.
        The layer norm here can select Layer Normalization type.
    Args:
        dim (int): Channel number of inputs.
        LayerNorm_type (str): Layer Normalization type.
    """

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(BaseModule):
    """Gated-Dconv Feed-Forward Network (GDFN)

    The original version of GDFN in
    "Restormer: Efficient Transformer for High-Resolution Image Restoration".

    Args:
        dim (int): Channel number of inputs.
        ffn_expansion_factor (float): channel expansion factor. Default: 2.66
        bias (bool): The bias of convolution.
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(BaseModule):
    """Multi-DConv Head Transposed Self-Attention (MDTA)

    The original version of MDTA in
    "Restormer: Efficient Transformer for High-Resolution Image Restoration".

    Args:
        dim (int): Channel number of inputs.
        num_heads (int): Number of attention heads.
        bias (bool): The bias of convolution.
    """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Forward results.
        """
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(
            q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(
            k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(
            v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(
            out,
            'b head c (h w) -> b (head c) h w',
            head=self.num_heads,
            h=h,
            w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(BaseModule):
    """Transformer Block.

    The original version of Transformer Block in "Restormer: Efficient\
        Transformer for High-Resolution Image Restoration".

    Args:
        dim (int): Channel number of inputs.
        num_heads (int): Number of attention heads.
        ffn_expansion_factor (float): channel expansion factor. Default: 2.66
        bias (bool): The bias of convolution.
        LayerNorm_type (str): Layer Normalization type.
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tesnor: Forward results.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(BaseModule):
    """Overlapped image patch embedding with 3x3 Conv.

    Args:
        in_c (int, optional): Channel number of inputs. Default: 3
        embed_dim (int, optional): embedding dimension. Default: 48
        bias (bool, optional): The bias of convolution. Default: False
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tesnor: Forward results.
        """
        x = self.proj(x)

        return x


class Downsample(BaseModule):
    """Downsample modules.

    Args:
        n_feat(int): Channel number of features.
    """

    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,
                n_feat // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tesnor: Forward results.
        """
        return self.body(x)


class Upsample(BaseModule):
    """Upsample modules.

    Args:
        n_feat(int): Channel number of features.
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,
                n_feat * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tesnor: Forward results.
        """
        return self.body(x)


@MODELS.register_module()
class Restormer(BaseModule):
    """Restormer A PyTorch impl of: `Restormer: Efficient Transformer for High-
    Resolution Image Restoration`. Ref repo:
    https://github.com/swz30/Restormer.

    Args:
        inp_channels (int): Number of input image channels. Default: 3.
        out_channels (int): Number of output image channels: 3.
        dim (int): Number of feature dimension. Default: 48.
        num_blocks (List(int)): Depth of each Transformer layer.
            Default: [4, 6, 6, 8].
        num_refinement_blocks (int): Number of refinement blocks.
            Default: 4.
        heads (List(int)): Number of attention heads in different layers.
            Default: 7.
        ffn_expansion_factor (float): Ratio of feed forward network expansion.
            Default: 2.66.
        bias (bool): The bias of convolution. Default: False
        LayerNorm_type (str|optional): Select layer Normalization type.
            Optional: 'WithBias','BiasFree'
            Default: 'WithBias'.
        dual_pixel_task (bool): True for dual-pixel defocus deblurring only.
            Also set inp_channels=6. Default: False.
        dual_keys (List): Keys of dual images in inputs.
            Default: ['imgL', 'imgR'].
    """

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 dual_keys=['imgL', 'imgR']):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type)
            for i in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        self.dual_keys = dual_keys
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(
                dim, int(dim * 2**1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(
            int(dim * 2**1),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)

    def forward(self, inp_img):
        """Forward function.

        Args:
            inp_img (Tensor): Input tensor with shape (B, C, H, W).
        Returns:
            Tensor: Forward results.
        """

        if self.dual_pixel_task:
            dual_images = [inp_img[key] for key in self.dual_keys]
            inp_img = torch.cat(dual_images, dim=1)

        _, _, h, w = inp_img.shape
        if h % 8 == 0:
            padding_h = 0
        else:
            padding_h = 8 - h % 8
        if w % 8 == 0:
            padding_w = 0
        else:
            padding_w = 8 - w % 8

        inp_img = F.pad(inp_img, (0, padding_w, 0, padding_h), 'reflect')
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1[:, :, :h, :w]
