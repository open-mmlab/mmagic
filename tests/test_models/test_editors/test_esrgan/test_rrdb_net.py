# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

from mmagic.models import RRDBNet


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
    input_shape = (1, 3, 12, 12)
    img = torch.rand(input_shape)
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
    input_shape = (1, 3, 12, 12)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 24, 24)

    # model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 24, 24)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
