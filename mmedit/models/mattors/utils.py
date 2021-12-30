# Copyright (c) OpenMMLab. All rights reserved.
import torch


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


def fba_fusion(alpha, img, F, B):
    """Postprocess the predicted.

    This class is adopted from
    https://github.com/MarcoForte/FBA_Matting.

    Args:
        alpha (Tensor): Tensor with shape (N, 1, H, W).
        img (Tensor): Tensor with shape (N, 3, H, W).
        F (Tensor): Tensor with shape (N, 3, H, W).
        B (Tensor): Tensor with shape (N, 3, H, W).

    Returns:
        alpha (Tensor): Tensor with shape (N, 1, H, W).
        F (Tensor): Tensor with shape (N, 3, H, W).
        B (Tensor): Tensor with shape (N, 3, H, W).
    """
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha *
         (1 - alpha) * F)

    F = torch.clamp(F, 0, 1)
    B = torch.clamp(B, 0, 1)
    la = 0.1
    alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (
        torch.sum((F - B) * (F - B), 1, keepdim=True) + la)
    alpha = torch.clamp(alpha, 0, 1)
    return alpha, F, B
