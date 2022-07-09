# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.image_restorers import TTSRDiscriminator


def test_ttsr_dict():
    net = TTSRDiscriminator(in_channels=3, in_size=160)
    # cpu
    inputs = torch.rand((2, 3, 160, 160))
    output = net(inputs)
    assert output.shape == (2, 1)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(inputs.cuda())
        assert output.shape == (2, 1)
