import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicUpsamplingFilter(nn.Module):
    """Dynamic upsampling filter used in DUF.

    Ref: https://github.com/yhjo09/VSR-DUF.
    It only supports input with 3 channels. And it applies the same filters
    to 3 channels.

    Args:
        filter_size (tuple): Filter size of generated filters.
            The shape is (kh, kw). Default: (5, 5).
    """

    def __init__(self, filter_size=(5, 5)):
        super(DynamicUpsamplingFilter, self).__init__()
        if not isinstance(filter_size, tuple):
            raise TypeError('The type of filter_size must be tuple, '
                            f'but got type{filter_size}')
        if len(filter_size) != 2:
            raise ValueError('The length of filter size must be 2, '
                             f'but got {len(filter_size)}.')
        # generate a local expansion filter, similar to im2col
        self.filter_size = filter_size
        filter_prod = np.prod(filter_size)
        expansion_filter = torch.eye(int(filter_prod)).view(
            filter_prod, 1, *filter_size)  # (kh*kw, 1, kh, kw)
        self.expansion_filter = expansion_filter.repeat(
            3, 1, 1, 1)  # repeat for all the 3 channels

    def forward(self, x, filters):
        """Forward function for DynamicUpsamplingFilter.

        Args:
            x (Tensor): Input image with 3 channels. The shape is (n, 3, h, w).
            filters (Tensor): Generated dynamic filters.
                The shape is (n, filter_prod, upsampling_square, h, w).
                filter_prod: prod of filter kenrel size, e.g., 1*5*5=25.
                upsampling_square: similar to pixel shuffle,
                    upsampling_square = upsampling * upsampling
                    e.g., for x 4 upsampling, upsampling_square= 4*4 = 16

        Returns:
            Tensor: Filtered image with shape (n, 3*upsampling, h, w)
        """
        n, filter_prod, upsampling_square, h, w = filters.size()
        kh, kw = self.filter_size
        expanded_input = F.conv2d(
            x,
            self.expansion_filter.to(x),
            padding=(kh // 2, kw // 2),
            groups=3)  # (n, 3*filter_prod, h, w)
        expanded_input = expanded_input.view(n, 3, filter_prod, h, w).permute(
            0, 3, 4, 1, 2)  # (n, h, w, 3, filter_prod)
        filters = filters.permute(
            0, 3, 4, 1, 2)  # (n, h, w, filter_prod, upsampling_square]
        out = torch.matmul(expanded_input,
                           filters)  # (n, h, w, 3, upsampling_square)
        return out.permute(0, 3, 4, 1, 2).view(n, 3 * upsampling_square, h, w)
