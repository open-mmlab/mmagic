# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models import SRCNNNet


def test_srcnn():
    # model, initialization and forward (cpu)
    net = SRCNNNet(
        channels=(3, 4, 6, 3), kernel_sizes=(9, 1, 5), upscale_factor=4)
    img = torch.rand(1, 3, 4, 4)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 16, 16)

    net = SRCNNNet(
        channels=(1, 4, 8, 1), kernel_sizes=(3, 3, 3), upscale_factor=2)
    img = torch.rand(1, 1, 4, 4)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1, 8, 8)

    # model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1, 8, 8)

    with pytest.raises(AssertionError):
        # The length of channel tuple should be 4
        net = SRCNNNet(
            channels=(3, 4, 3), kernel_sizes=(9, 1, 5), upscale_factor=4)
    with pytest.raises(AssertionError):
        # The length of kernel tuple should be 3
        net = SRCNNNet(
            channels=(3, 4, 4, 3), kernel_sizes=(9, 1, 1, 5), upscale_factor=4)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
