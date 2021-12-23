def pixel_unshuffle(x, scale):
    """Down-sample by pixel unshuffle.

    Args:
        x (Tensor): Input tensor.
        scale (int): Scale factor.

    Returns:
        y (Tensor): Output tensor.
    """

    b, c, h, w = x.shape
    h = int(h / scale)
    w = int(w / scale)
    x = x.view(b, c, h, scale, w, scale)
    shuffle_out = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    y = shuffle_out.view(b, c * scale * scale, h, w)
    return y
