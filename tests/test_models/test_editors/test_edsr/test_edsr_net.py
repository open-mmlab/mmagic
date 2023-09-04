# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors import EDSRNet


def test_edsr_cpu():
    """Test EDSRNet."""

    # x2 model
    EDSRNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=2)
    # x3 model, initialization and forward (cpu)
    net = EDSRNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=3)
    img = torch.rand(1, 3, 12, 12)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 36, 36)
    # x4 modeland, initialization and forward (cpu)
    net = EDSRNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=4)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 48, 48)
    # gray x4 modeland, initialization and forward (cpu)
    net = EDSRNet(
        in_channels=1,
        out_channels=1,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=4,
        rgb_mean=[0],
        rgb_std=[1])
    gray = torch.rand((1, 1, 12, 12))
    output = net(gray)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1, 48, 48)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_edsr_cuda():

    net = EDSRNet(
        in_channels=1,
        out_channels=1,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=4,
        rgb_mean=[0],
        rgb_std=[1])
    gray = torch.rand((1, 1, 12, 12))

    # x4 model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(gray.cuda())
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1, 48, 48)

    with pytest.raises(ValueError):
        # Currently supported upscale_factor is 2^n and 3
        EDSRNet(
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=5)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
