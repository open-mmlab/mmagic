# Copyright (c) OpenMMLab. All rights reserved.
def pixel_unshuffle(x, scale):
    """Down-sample by pixel unshuffle.

    Args:
        x (Tensor): Input tensor.
        scale (int): Scale factor.

    Returns:
        Tensor: Output tensor.
    """

    b, c, h, w = x.shape
    if h % scale != 0 or w % scale != 0:
        raise AssertionError(
            f'Invalid scale ({scale}) of pixel unshuffle for tensor '
            f'with shape: {x.shape}')
    h = int(h / scale)
    w = int(w / scale)
    x = x.view(b, c, h, scale, w, scale)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, -1, h, w)
