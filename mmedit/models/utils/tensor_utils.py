# Copyright (c) OpenMMLab. All rights reserved.
import torch


def get_unknown_tensor(trimap, unknown_value=128 / 255):
    """Get 1-channel unknown area tensor from the 3 or 1-channel trimap tensor.

    Args:
        trimap (Tensor): Tensor with shape (N, 3, H, W) or (N, 1, H, W).
        unknown_value (float): Scalar value indicating unknown region in
            trimap.
            If trimap is pre-processed using `'rescale_to_zero_one'`, then
            0 for bg, 128/255 for unknown, 1 for fg,
            and unknown_value should set to 128 / 255.
            If trimap is pre-processed by
            :meth:`FormatTrimap(to_onehot=False)`, then
            0 for bg, 1 for unknown, 2 for fg
            and unknown_value should set to 1.
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


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize vector with it's lengths at the last dimension. If `vector` is
    two-dimension tensor, this function is same as L2 normalization.

    Args:
        vector (torch.Tensor): Vectors to be normalized.

    Returns:
        torch.Tensor: Vectors after normalization.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def truncated_normal(tensor, mean=0, std=1, n_truncted_stds=2):
    assert std >= 0, f"{std}"
    size = tensor.shape

    # [n, 1] -> [n, 1, 4]
    tmp = tensor.new_empty(size + (4,), device=tensor.device).normal_()
    tmp.data.mul_(std).add_(mean)

    lower_bound = mean - 1 * n_truncted_stds * std
    upper_bound = mean + n_truncted_stds * std
    valid = (tmp < upper_bound) & (tmp > lower_bound)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))

    try:
        assert torch.all(tensor >= lower_bound), f"{torch.min(tensor)}"
        assert torch.all(tensor <= upper_bound), f"{torch.max(tensor)}"
    except:
        # fmt: off
        print("\nin truncated normal lower bound: ", tensor.shape, lower_bound, torch.min(tensor), torch.sum(tensor >= lower_bound))
        print("\nin truncated normal upper bound: ", tensor.shape, upper_bound, torch.max(tensor), torch.sum(tensor <= lower_bound))
        tensor[tensor <= lower_bound] = lower_bound
        tensor[tensor >= upper_bound] = upper_bound
        # fmt: on

    return tensor
