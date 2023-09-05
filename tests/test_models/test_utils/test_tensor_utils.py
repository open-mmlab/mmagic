# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.utils.tensor_utils import get_unknown_tensor, normalize_vecs


def test_tensor_utils():
    input = torch.rand((2, 3, 128, 128))
    output = get_unknown_tensor(input)
    assert output.detach().numpy().shape == (2, 1, 128, 128)
    input = torch.rand((2, 1, 128, 128))
    output = get_unknown_tensor(input)
    assert output.detach().numpy().shape == (2, 1, 128, 128)


def test_normalize_vector():
    inp = torch.randn(4, 10)
    out = normalize_vecs(inp)

    def l2_norm(a):
        return a / torch.norm(a, dim=1, keepdim=True)

    l2_norm_out = l2_norm(inp)

    assert (out == l2_norm_out).all()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
