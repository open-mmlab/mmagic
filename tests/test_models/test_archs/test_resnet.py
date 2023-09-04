# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.archs import ResNet


def test_resnet():
    resnet = ResNet(18, 3, 16, 16)
    resnet.init_weights()
    input = torch.rand((2, 3, 128, 128))
    output = resnet(input)
    assert output[0].detach().numpy().shape == (2, 3, 128, 128)

    resnet = ResNet(50, 3, 16, 16)
    resnet.init_weights()
    input = torch.rand((2, 3, 128, 128))
    output = resnet(input)
    assert output[0].detach().numpy().shape == (2, 3, 128, 128)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
