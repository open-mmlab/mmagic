# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.builder import build_component


def test_search_transformer():
    model_cfg = dict(type='SearchTransformer')
    model = build_component(model_cfg)

    lr_pad_level3 = torch.randn((2, 32, 32, 32))
    ref_pad_level3 = torch.randn((2, 32, 32, 32))
    ref_level3 = torch.randn((2, 32, 32, 32))
    ref_level2 = torch.randn((2, 16, 64, 64))
    ref_level1 = torch.randn((2, 8, 128, 128))

    s, textures = model(lr_pad_level3, ref_pad_level3,
                        (ref_level3, ref_level2, ref_level1))
    t_level3, t_level2, t_level1 = textures

    assert s.shape == (2, 1, 32, 32)
    assert t_level3.shape == (2, 32, 32, 32)
    assert t_level2.shape == (2, 16, 64, 64)
    assert t_level1.shape == (2, 8, 128, 128)
