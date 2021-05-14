import torch

from mmedit.models import build_component


def test_search_transformer():
    model_cfg = dict(type='SearchTransformer')
    model = build_component(model_cfg)

    lr_pad_lv3 = torch.randn((2, 32, 32, 32))
    ref_pad_lv3 = torch.randn((2, 32, 32, 32))
    ref_lv3 = torch.randn((2, 32, 32, 32))
    ref_lv2 = torch.randn((2, 16, 64, 64))
    ref_lv1 = torch.randn((2, 8, 128, 128))

    s, t_lv3, t_lv2, t_lv1 = model(lr_pad_lv3, ref_pad_lv3, ref_lv1, ref_lv2,
                                   ref_lv3)

    assert s.shape == (2, 1, 32, 32)
    assert t_lv3.shape == (2, 32, 32, 32)
    assert t_lv2.shape == (2, 16, 64, 64)
    assert t_lv1.shape == (2, 8, 128, 128)
