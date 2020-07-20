import pytest
import torch
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmedit.models.backbones import PConvEncoder, PConvEncoderDecoder


def test_pconv_encdec():
    pconv_enc_cfg = dict(type='PConvEncoder')
    pconv_dec_cfg = dict(type='PConvDecoder')

    if torch.cuda.is_available():
        pconv_encdec = PConvEncoderDecoder(pconv_enc_cfg, pconv_dec_cfg)
        pconv_encdec.init_weights()
        pconv_encdec.cuda()
        x = torch.randn((1, 3, 256, 256)).cuda()
        mask = torch.ones_like(x)
        mask[..., 50:150, 100:250] = 1.
        res, updated_mask = pconv_encdec(x, mask)
        assert res.shape == (1, 3, 256, 256)
        assert mask.shape == (1, 3, 256, 256)

        with pytest.raises(TypeError):
            pconv_encdec.init_weights(pretrained=dict(igccc=8989))


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
