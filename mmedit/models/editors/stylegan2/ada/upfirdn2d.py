# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.ops.upfirdn2d import upfirdn2d


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    fw = int(fw)
    fh = int(fh)
    assert fw >= 1 and fh >= 1
    return fw, fh


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.
    By default, the result is padded so that its shape is a multiple of the
    input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.
    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a
                     list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single
                     number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before,
                     y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'`
                     (default: `'cuda'`).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]

    gain = gain * upx * upy
    f = f * (gain**(f.ndim / 2))
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    if f.ndim == 1:
        x = upfirdn2d(x, f.unsqueeze(0), up=(upx, 1), pad=(p[0], p[1], 0, 0))
        x = upfirdn2d(x, f.unsqueeze(1), up=(1, upy), pad=(0, 0, p[2], p[3]))
    return x


def setup_filter(f,
                 device=torch.device('cpu'),
                 normalize=True,
                 flip_filter=False,
                 gain=1,
                 separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.
    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically)
    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain**(f.ndim / 2))
    f = f.to(device=device)
    return f


def downsample2d(x,
                 f,
                 down=2,
                 padding=0,
                 flip_filter=False,
                 gain=1,
                 impl='cuda'):
    r"""Downsample a batch of 2D images using the given 2D FIR filter.
    By default, the result is padded so that its shape is a fraction of the
    input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.
    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a
                     list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number
                     or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before,
                     y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'`
                     (default: `'cuda'`).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    if f.ndim == 1:
        x = upfirdn2d(
            x, f.unsqueeze(0), down=(downx, 1), pad=(p[0], p[1], 0, 0))
        x = upfirdn2d(
            x, f.unsqueeze(1), down=(1, downy), pad=(0, 0, p[2], p[3]))
    return x
