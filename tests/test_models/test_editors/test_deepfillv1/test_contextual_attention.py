# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors import ContextualAttentionModule


def test_deepfill_contextual_attention_module():
    cmodule = ContextualAttentionModule()
    x = torch.rand((2, 128, 64, 64))
    mask = torch.zeros((2, 1, 64, 64))
    mask[..., 20:100, 23:90] = 1.
    res, offset = cmodule(x, x, mask)
    assert res.shape == (2, 128, 64, 64)
    assert offset.shape == (2, 32, 32, 32, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
