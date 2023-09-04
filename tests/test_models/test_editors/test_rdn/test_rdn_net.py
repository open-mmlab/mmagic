# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch.nn as nn

from mmagic.models import RDNNet


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_rdn():

    scale = 4

    model = RDNNet(
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        channel_growth=32,
        num_blocks=16,
        upscale_factor=scale)

    # test attributes
    assert model.__class__.__name__ == 'RDNNet'

    # prepare data
    inputs = torch.rand(1, 3, 32, 16)
    targets = torch.rand(1, 3, 128, 64)

    # prepare loss
    loss_function = nn.L1Loss()

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # test on cpu
    output = model(inputs)
    optimizer.zero_grad()
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs)
        optimizer.zero_grad()
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()
        assert torch.is_tensor(output)
        assert output.shape == targets.shape


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
