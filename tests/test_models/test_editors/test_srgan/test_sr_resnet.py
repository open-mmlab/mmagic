# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models import ModifiedVGG, MSRResNet


def test_srresnet_backbone():
    """Test SRResNet backbone."""

    # x2 model
    MSRResNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=2)
    # x3 model, initialization and forward (cpu)
    net = MSRResNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=3)
    input_shape = (1, 3, 12, 12)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 36, 36)
    # x4 modeland, initialization and forward (cpu)
    net = MSRResNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=4)
    output = net(img)
    assert output.shape == (1, 3, 48, 48)

    # x4 model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 48, 48)

    with pytest.raises(ValueError):
        # Currently supported upscale_factor is [2, 3, 4]
        MSRResNet(
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=16)


def test_discriminator():
    """Test discriminator backbone."""

    # model, initialization and forward (cpu)
    net = ModifiedVGG(in_channels=3, mid_channels=64)
    input_shape = (1, 3, 128, 128)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 1)

    # model, initialization and forward (gpu)
    if torch.cuda.is_available():
        net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1)

    with pytest.raises(AssertionError):
        # input size must be 128 * 128
        input_shape = (1, 3, 64, 64)
        img = torch.rand(input_shape)
        output = net(img)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
