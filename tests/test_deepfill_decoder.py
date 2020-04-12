import torch
from mmedit.models.backbones import DeepFillDecoder


def test_deepfill_dec():

    decoder = DeepFillDecoder(128, out_act_cfg=None)
    assert not decoder.with_out_activation

    decoder = DeepFillDecoder(128)
    x = torch.randn((2, 128, 64, 64))
    input_dict = dict(out=x)
    res = decoder(input_dict)

    assert res.shape == (2, 3, 256, 256)
    assert decoder.dec2.stride == (1, 1)
    assert decoder.dec2.out_channels == 128
    assert not decoder.dec7.with_activation
    assert res.min().item() >= -1. and res.max().item() <= 1
    if torch.cuda.is_available():
        decoder = DeepFillDecoder(128).cuda()
        x = torch.randn((2, 128, 64, 64)).cuda()
        input_dict = dict(out=x)
        res = decoder(input_dict)
        assert res.shape == (2, 3, 256, 256)
        assert decoder.dec2.stride == (1, 1)
        assert decoder.dec2.out_channels == 128
        assert not decoder.dec7.with_activation
        assert res.min().item() >= -1. and res.max().item() <= 1
