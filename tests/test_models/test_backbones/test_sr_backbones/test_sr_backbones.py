# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.models.backbones import MSRResNet, RRDBNet
from mmedit.models.components import ModifiedVGG


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
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 12, 12)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 36, 36)
    # x4 modeland, initialization and forward (cpu)
    net = MSRResNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=4)
    net.init_weights(pretrained=None)
    output = net(img)
    assert output.shape == (1, 3, 48, 48)

    # x4 model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 48, 48)

    with pytest.raises(TypeError):
        # pretrained should be str or None
        net.init_weights(pretrained=[1])

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
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 128, 128)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 1)

    # model, initialization and forward (gpu)
    if torch.cuda.is_available():
        net.init_weights(pretrained=None)
        net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1)

    with pytest.raises(TypeError):
        # pretrained should be str or None
        net.init_weights(pretrained=[1])

    with pytest.raises(AssertionError):
        # input size must be 128 * 128
        input_shape = (1, 3, 64, 64)
        img = _demo_inputs(input_shape)
        output = net(img)


def test_rrdbnet_backbone():
    """Test RRDBNet backbone."""

    # model, initialization and forward (cpu)
    # x4 model
    net = RRDBNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        growth_channels=4,
        upscale_factor=4)
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 12, 12)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 48, 48)

    # x3 model
    with pytest.raises(ValueError):
        net = RRDBNet(
            in_channels=3,
            out_channels=3,
            mid_channels=8,
            num_blocks=2,
            growth_channels=4,
            upscale_factor=3)

    # x2 model
    net = RRDBNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        growth_channels=4,
        upscale_factor=2)
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 12, 12)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 24, 24)

    # model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 24, 24)

    with pytest.raises(TypeError):
        # pretrained should be str or None
        net.init_weights(pretrained=[1])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).

    Returns:
        imgs: (Tensor): Images in FloatTensor with desired shapes.
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs
