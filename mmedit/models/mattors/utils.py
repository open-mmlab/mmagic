def get_unknown_tensor(trimap, meta):
    """Get 1-channel unknown area tensor from the 3 or 1-channel trimap tensor.

    Args:
        trimap (Tensor): Tensor with shape (N, 3, H, W) or (N, 1, H, W).

    Returns:
        Tensor: Unknown area mask of shape (N, 1, H, W).
    """
    if trimap.shape[1] == 3:
        # The three channels correspond to (bg mask, unknown mask, fg mask)
        # respectively.
        weight = trimap[:, 1:2, :, :].float()
    elif 'to_onehot' in meta[0]:
        # key 'to_onehot' is added by pipeline `FormatTrimap`
        # 0 for bg, 1 for unknown, 2 for fg
        weight = trimap.eq(1).float()
    else:
        # trimap is simply processed by pipeline `RescaleToZeroOne`
        # 0 for bg, 128/255 for unknown, 1 for fg
        weight = trimap.eq(128 / 255).float()
    return weight
