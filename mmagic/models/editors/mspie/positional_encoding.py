# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.model import BaseModule

from mmagic.registry import MODELS


@MODELS.register_module('SPE')
@MODELS.register_module('SPE2d')
class SinusoidalPositionalEmbedding(BaseModule):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).

    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa

    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encoding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.

    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].

        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        """Make position tensors.

        Args:
            input (tensor): Input tensor.
            padding_idx (int | list[int]): The index for the padding contents.
            The padding positions will obtain an encoding vector filling
            in zeros.

        Returns:
            tensor: Position tensors.
        """
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        """Make 2-d grid mask.

        Args:
            height (int): Height of the grid.
            width (int): Width of the grid.
            num_batches (int, optional): The number of batch size.
                Defaults to 1.
            center_shift (int | None, optional): Shift the center point to some
                index. Defaults to None.

        Returns:
            Tensor: 2-d Grid mask.
        """
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)

        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]

        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


@MODELS.register_module('CSG2d')
@MODELS.register_module('CSG')
@MODELS.register_module()
class CatersianGrid(BaseModule):
    """Catersian Grid for 2d tensor.

    The Catersian Grid is a common-used positional encoding in deep learning.
    In this implementation, we follow the convention of ``grid_sample`` in
    PyTorch. In other words, ``[-1, -1]`` denotes the left-top corner while
    ``[1, 1]`` denotes the right-bottom corner.
    """

    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    def make_grid2d(self, height, width, num_batches=1, requires_grad=False):
        h, w = height, width
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = torch.stack((grid_x, grid_y), 0)
        grid.requires_grad = requires_grad

        grid = torch.unsqueeze(grid, 0)
        grid = grid.repeat(num_batches, 1, 1, 1)

        return grid

    def make_grid2d_like(self, x, requires_grad=False):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)

        Note that the positional embedding highly depends on the the function,
        ``make_grid2d``.
        """
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)
