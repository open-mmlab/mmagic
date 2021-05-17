import torch

from mmedit.models.backbones.sr_backbones.ttsr_net import CSFI2, CSFI3, SFE


def test_sfe():
    inputs = torch.rand(2, 3, 48, 48)
    sfe = SFE(3, 64, 16, 1.)
    outputs = sfe(inputs)
    assert outputs.shape == (2, 64, 48, 48)


def test_csfi():
    inputs1 = torch.rand(2, 16, 24, 24)
    inputs2 = torch.rand(2, 16, 48, 48)
    inputs3 = torch.rand(2, 16, 96, 96)

    csfi2 = CSFI2(mid_channels=16)
    out1, out2 = csfi2(inputs1, inputs2)
    assert out1.shape == (2, 16, 24, 24)
    assert out2.shape == (2, 16, 48, 48)

    csfi3 = CSFI3(mid_channels=16)
    out1, out2, out3 = csfi3(inputs1, inputs2, inputs3)
    assert out1.shape == (2, 16, 24, 24)
    assert out2.shape == (2, 16, 48, 48)
    assert out3.shape == (2, 16, 96, 96)


if __name__ == '__main__':
    test_sfe()
    test_csfi()
