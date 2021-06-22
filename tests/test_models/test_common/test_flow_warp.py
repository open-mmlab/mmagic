import pytest
import torch

from mmedit.models import flow_warp


def tensor_shift(x, shift=(1, 1), fill_val=0):
    """Shift tensor for testing flow_warp.

    Args:
        x (Tensor): the input tensor. The shape is (b, c, h, w].
        shift (tuple): shift pixel.
        fill_val (float): fill value.

    Returns:
        Tensor: the shifted tensor.
    """

    _, _, h, w = x.size()
    shift_h, shift_w = shift
    new = torch.ones_like(x) * fill_val

    len_h = h - shift_h
    len_w = w - shift_w
    new[:, :, shift_h:shift_h + len_h,
        shift_w:shift_w + len_w] = x.narrow(2, 0, len_h).narrow(3, 0, len_w)
    return new


def test_flow_warp():
    x = torch.rand(1, 3, 10, 10)
    flow = torch.rand(1, 4, 4, 2)
    with pytest.raises(ValueError):
        # The spatial sizes of input and flow are not the same.
        flow_warp(x, flow)

    # cpu
    x = torch.rand(1, 3, 10, 10)
    flow = -torch.ones(1, 10, 10, 2)
    result = flow_warp(x, flow)
    assert result.size() == (1, 3, 10, 10)
    error = torch.sum(torch.abs(result - tensor_shift(x, (1, 1))))
    assert error < 1e-5
    # gpu
    if torch.cuda.is_available():
        x = torch.rand(1, 3, 10, 10).cuda()
        flow = -torch.ones(1, 10, 10, 2).cuda()
        result = flow_warp(x, flow)
        assert result.size() == (1, 3, 10, 10)
        error = torch.sum(torch.abs(result - tensor_shift(x, (1, 1))))
        assert error < 1e-5
