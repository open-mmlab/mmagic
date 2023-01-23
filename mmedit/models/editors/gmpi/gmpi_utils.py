# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.functional import conv2d, conv_transpose2d

from mmedit.models.editors.stylegan3.stylegan3_ops.ops import upfirdn2d
from mmedit.models.editors.stylegan3.stylegan3_ops.ops.upfirdn2d import (
    _get_filter_size, _parse_padding)


def transform_vectors(matrix: torch.Tensor,
                      vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize vector lengths."""
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """Dot product of two tensors."""
    return (x * y).sum(-1)


def fma(a, b, c):  # => a * b + c
    return _FusedMultiplyAdd.apply(a, b, c)


class _FusedMultiplyAdd(torch.autograd.Function):  # a * b + c

    @staticmethod
    def forward(ctx, a, b, c):  # pylint: disable=arguments-differ
        out = torch.addcmul(c, a, b)
        ctx.save_for_backward(a, b)
        ctx.c_shape = c.shape
        return out

    @staticmethod
    def backward(ctx, d_out):  # pylint: disable=arguments-differ
        a, b = ctx.saved_tensors
        c_shape = ctx.c_shape
        da = None
        db = None
        dc = None

        if ctx.needs_input_grad[0]:
            da = _unbroadcast(d_out * b, a.shape)

        if ctx.needs_input_grad[1]:
            db = _unbroadcast(d_out * a, b.shape)

        if ctx.needs_input_grad[2]:
            dc = _unbroadcast(d_out, c_shape)

        return da, db, dc


def _unbroadcast(x, shape):
    extra_dims = x.ndim - len(shape)
    assert extra_dims >= 0
    dim = [
        i for i in range(x.ndim)
        if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)
    ]
    if len(dim):
        x = x.sum(dim=dim, keepdim=True)
    if extra_dims:
        x = x.reshape(-1, *x.shape[extra_dims + 1:])
    assert x.shape == shape
    return x


def _get_weight_shape(w):
    shape = [int(sz) for sz in w.shape]
    return shape


def _conv2d_wrapper(x,
                    w,
                    stride=1,
                    padding=0,
                    groups=1,
                    transpose=False,
                    flip_weight=True):
    """Wrapper for the underlying `conv2d()` and `conv_transpose2d()`
    implementations."""
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)

    # Flip weight if requested.
    # conv2d() actually performs correlation (flip_weight=True)
    # not convolution (flip_weight=False).
    if not flip_weight:
        w = w.flip([2, 3])

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if kw == 1 and kh == 1 and stride == 1 and padding in [0, [0, 0], (0, 0)
                                                           ] and not transpose:
        if x.stride()[1] == 1 and min(out_channels,
                                      in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape(
                    [in_shape[0], in_channels_per_group, -1])
                x = x.reshape(
                    [in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                x = x.to(memory_format=torch.contiguous_format)
                w = w.to(memory_format=torch.contiguous_format)
                x = torch.nn.functional.conv2d(
                    input=x,
                    weight=w,
                    bias=None,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=groups)
            return x.to(memory_format=torch.channels_last)

    # Otherwise => execute using conv2d.
    op = conv_transpose2d if transpose else conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)


def conv2d_resample(x,
                    w,
                    f=None,
                    up=1,
                    down=1,
                    padding=0,
                    groups=1,
                    flip_weight=True,
                    flip_filter=False):
    r"""2D convolution with optional up/downsampling.

    Padding is performed only once at the beginning,
    not between the operations.

    Args:
        x:      Input tensor of shape
                `[batch_size, in_channels, in_height, in_width]`.
        w:      Weight tensor of shape
                `[out_channels, in_channels//groups,
                    kernel_height, kernel_width]`.
        f:      Low-pass filter for up/downsampling. Must be prepared
                beforehand by calling upfirdn2d.setup_filter().
                None = identity (default).
        up:     Integer upsampling factor (default: 1).
        down:   Integer downsampling factor (default: 1).
        padding:Padding with respect to the upsampled image. Can be a single
                number or a list/tuple `[x, y]` or
                `[x_before, x_after, y_before, y_after]` (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation
                        (default: True).
        flip_filter:    False = convolution, True = correlation
                        (default: False).

    Returns:
        Tensor of the shape
        `[batch_size, num_channels, out_height, out_width]`.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and (x.ndim == 4)
    assert isinstance(w, torch.Tensor) and (w.ndim == 4) and (w.dtype
                                                              == x.dtype)
    assert f is None or (isinstance(f, torch.Tensor) and f.ndim in [1, 2]
                         and f.dtype == torch.float32)
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    assert isinstance(groups, int) and (groups >= 1)
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)
    fw, fh = _get_filter_size(f)
    px0, px1, py0, py1 = _parse_padding(padding)

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first
    # then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d.upfirdn2d(
            x=x,
            f=f,
            down=down,
            padding=[px0, px1, py0, py1],
            flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first
    # then upsample.
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d.upfirdn2d(
            x=x,
            f=f,
            up=up,
            padding=[px0, px1, py0, py1],
            gain=up**2,
            flip_filter=flip_filter)
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d.upfirdn2d(
            x=x, f=f, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(
            x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: upsampling with optional downsampling
    # => use transpose strided convolution.
    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups,
                          in_channels_per_group, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(groups * in_channels_per_group,
                          out_channels // groups, kh, kw)
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(
            x=x,
            w=w,
            stride=up,
            padding=[pyt, pxt],
            groups=groups,
            transpose=True,
            flip_weight=(not flip_weight))
        x = upfirdn2d.upfirdn2d(
            x=x,
            f=f,
            padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt],
            gain=up**2,
            flip_filter=flip_filter)
        if down > 1:
            x = upfirdn2d.upfirdn2d(
                x=x, f=f, down=down, flip_filter=flip_filter)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying
    # implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(
                x=x,
                w=w,
                padding=[py0, px0],
                groups=groups,
                flip_weight=flip_weight)

    # Fallback: Generic reference implementation.
    x = upfirdn2d.upfirdn2d(
        x=x,
        f=(f if up > 1 else None),
        up=up,
        padding=[px0, px1, py0, py1],
        gain=up**2,
        flip_filter=flip_filter)
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x
