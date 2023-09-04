# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmagic.models.editors import PConvEncoder


def test_pconv_enc():
    pconv_enc = PConvEncoder(norm_eval=False)
    pconv_enc.train()
    for name, module in pconv_enc.named_modules():
        if isinstance(module, _BatchNorm):
            assert module.training

    pconv_enc = PConvEncoder(norm_eval=True)
    pconv_enc.train()
    for name, module in pconv_enc.named_modules():
        if isinstance(module, _BatchNorm):
            assert not module.training

    pconv_enc = PConvEncoder()
    img = torch.randn(2, 3, 128, 128)
    mask = torch.ones_like(img)
    output = pconv_enc(img, mask)
    assert isinstance(output['hidden_feats'], dict)
    assert isinstance(output['hidden_masks'], dict)
    assert output['out'].detach().numpy().shape == (2, 512, 1, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
