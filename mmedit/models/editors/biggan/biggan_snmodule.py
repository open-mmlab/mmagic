# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

# yapf:disable
'''
    Ref: Functions in this file are borrowed from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py # noqa
'''
# yapf:enable


def proj(x, y):
    """Calculate Projection of x onto y.

    Args:
        x (torch.Tensor): Projection vector x.
        y (torch.Tensor): Direction vector y.

    Returns:
        torch.Tensor: Projection of x onto y.
    """
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    """Orthogonalize x w.r.t list of vectors ys.

    Args:
        x (torch.Tensor): Vector to be added into the
            orthogonal vectors.
        ys (list[torch.Tensor]): A set of orthogonal vectors.

    Returns:
        torch.Tensor: Result of Gram–Schmidt orthogonalization.
    """
    for y in ys:
        x = x - proj(x, y)
    return x


@torch.no_grad()
def power_iteration(weight, u_list, update=True, eps=1e-12):
    """Power iteration method for calculating spectral norm.

    Args:
        weight (torch.Tensor): Module weight.
        u_list (list[torch.Tensor]): list of left singular vector.
            The length of list equals to the simulation times.
        update (bool, optional): Whether update left singular
            vector. Defaults to True.
        eps (float, optional): Vector Normalization epsilon.
            Defaults to 1e-12.

    Returns:
        tuple[list[tensor.Tensor]]: Tuple consist of three lists
            which contain singular values, left singular
            vector and right singular vector respectively.
    """
    us, vs, svs = [], [], []
    for i, u in enumerate(u_list):
        v = torch.matmul(u, weight)
        v = F.normalize(gram_schmidt(v, vs), eps=eps)
        vs += [v]
        u = torch.matmul(v, weight.t())
        u = F.normalize(gram_schmidt(u, us), eps=eps)
        us += [u]
        if update:
            u_list[i][:] = u
        svs += [
            torch.squeeze(torch.matmul(torch.matmul(v, weight.t()), u.t()))
        ]
    return svs, us, vs


class SpectralNorm(object):
    """Spectral normalization base class.

    Args:
        num_svs (int): Number of singular values.
        num_iters (int): Number of power iterations per step.
        num_outputs (int): Number of output channels.
        transpose (bool, optional): If set to `True`, weight
            matrix will be transposed before power iteration.
            Defaults to False.
        eps (float, optional): Vector Normalization epsilon for
            avoiding divide by zero. Defaults to 1e-12.
    """

    def __init__(self,
                 num_svs,
                 num_iters,
                 num_outputs,
                 transpose=False,
                 eps=1e-12):
        self.num_iters = num_iters
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        """Get left singular vectors."""
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        """Get singular values."""
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def sn_weight(self):
        """Compute the spectrally-normalized weight."""
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_iters power iterations
        for _ in range(self.num_iters):
            svs, us, vs = power_iteration(
                W_mat, self.u, update=self.training, eps=self.eps)
        # Update the svs
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[-1]


class SNConv2d(nn.Conv2d, SpectralNorm):
    """2D Conv layer with spectral norm.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int, optional): Stride of the convolution.. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of
            the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements.
            Defaults to 1.
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Defaults to 1.
        bias (bool, optional): Whether to use bias parameter.
            Defaults to True.
        num_svs (int): Number of singular values.
        num_iters (int): Number of power iterations per step.
        eps (float, optional): Vector Normalization epsilon for
            avoiding divide by zero. Defaults to 1e-12.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 num_svs=1,
                 num_iters=1,
                 eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           stride, padding, dilation, groups, bias)
        SpectralNorm.__init__(self, num_svs, num_iters, out_channels, eps=eps)

    def forward(self, x):
        """Forward function."""
        return F.conv2d(x, self.sn_weight(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SpectralNorm):
    """Linear layer with spectral norm.

    Args:
        in_features (int): Number of channels in the input feature.
        out_features (int): Number of channels in the out feature.
        bias (bool, optional):  Whether to use bias parameter.
            Defaults to True.
        num_svs (int): Number of singular values.
        num_iters (int): Number of power iterations per step.
        eps (float, optional): Vector Normalization epsilon for
            avoiding divide by zero. Defaults to 1e-12.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 num_svs=1,
                 num_iters=1,
                 eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SpectralNorm.__init__(self, num_svs, num_iters, out_features, eps=eps)

    def forward(self, x):
        """Forward function."""
        return F.linear(x, self.sn_weight(), self.bias)


# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SpectralNorm):
    """Embedding layer with spectral norm.

    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (int, optional):  If specified, the entries at
            padding_idx do not contribute to the gradient; therefore,
            the embedding vector at padding_idx is not updated during
            training, i.e. it remains as a fixed “pad”. For a newly
            constructed Embedding, the embedding vector at padding_idx
            will default to all zeros, but can be updated to another value
            to be used as the padding vector. Defaults to None.
        max_norm (float, optional): If given, each embedding vector with
            norm larger than max_norm is renormalized to have norm
            max_norm. Defaults to None.
        norm_type (int, optional):  The p of the p-norm to compute for
            the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale
            gradients by the inverse of frequency of the words in the
            mini-batch. Default False.
        sparse (bool, optional):  If True, gradient w.r.t. weight matrix
            will be a sparse tensor. See Notes for more details regarding
            sparse gradients. Defaults to False.
        _weight (torch.Tensor, optional): Initial Weight. Defaults to None.
        num_svs (int): Number of singular values.
        num_iters (int): Number of power iterations per step.
        eps (float, optional): Vector Normalization epsilon for
            avoiding divide by zero. Defaults to 1e-12.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 num_svs=1,
                 num_iters=1,
                 eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                              max_norm, norm_type, scale_grad_by_freq, sparse,
                              _weight)
        SpectralNorm.__init__(
            self, num_svs, num_iters, num_embeddings, eps=eps)

    def forward(self, x):
        """Forward function."""
        return F.embedding(x, self.sn_weight())
