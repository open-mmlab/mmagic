import torch

from mmedit.models.backbones.sr_backbones.ttsr_net import SFE


def test_ttsr():
    inputs = torch.rand(2, 3, 48, 48)
    sfe = SFE(3, 64, 16, 1.)
    outputs = sfe(inputs)
    assert outputs.shape == (2, 64, 48, 48)


if __name__ == '__main__':
    test_ttsr()
