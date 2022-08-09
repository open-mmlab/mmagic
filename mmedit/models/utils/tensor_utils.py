# Copyright (c) OpenMMLab. All rights reserved.


def get_unknown_tensor(trimap, unknown_value=128 / 255):
    """Get 1-channel unknown area tensor from the 3 or 1-channel trimap tensor.

    Args:
        trimap (Tensor): Tensor with shape (N, 3, H, W) or (N, 1, H, W).
        unknown_value (float): Scalar value indicating unknown region in trimap
            If trimap is pre-processed using `'rescale_to_zero_one'`, then
                0 for bg, 128/255 for unknown, 1 for fg,
                and unknown_value should set to 128 / 255
            If trimap is pre-processed by
            :meth:`FormatTrimap(to_onehot=False)`, then
                0 for bg, 1 for unknown, 2 for fg
                and unknown_value should set to 1
            If trimap is pre-processed by
            :meth:`FormatTrimap(to_onehot=True)`, then
                trimap is 3-channeled, and this value is not used.

    Returns:
        Tensor: Unknown area mask of shape (N, 1, H, W).
    """
    if trimap.shape[1] == 3:
        # The three channels correspond to (bg mask, unknown mask, fg mask)
        # respectively.
        weight = trimap[:, 1:2, :, :].float()
    # elif 'to_onehot' in meta[0]:
    # key 'to_onehot' is added by pipeline `FormatTrimap`
    # 0 for bg, 1 for unknown, 2 for fg
    # weight = trimap.eq(1).float()
    else:
        # trimap is simply processed by pipeline `RescaleToZeroOne`
        # 0 for bg, 128/255 for unknown, 1 for fg
        weight = trimap.eq(unknown_value).float()
    return weight
