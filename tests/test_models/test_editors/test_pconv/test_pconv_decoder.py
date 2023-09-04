# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors import PConvDecoder, PConvEncoder
from mmagic.utils import register_all_modules

register_all_modules()


def test_pconv_dec():
    pconv_enc = PConvEncoder()
    img = torch.randn(2, 3, 128, 128)
    mask = torch.ones_like(img)
    output = pconv_enc(img, mask)

    pconv_dec = PConvDecoder()
    input = {
        'hidden_feats': output['hidden_feats'],
        'hidden_masks': output['hidden_masks']
    }
    h, h_mask = pconv_dec(input)
    assert h.detach().numpy().shape == (2, 3, 128, 128)
    assert h_mask.detach().numpy().shape == (2, 3, 128, 128)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
