# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.ifrnet.ifrnet_utils import resize, warp


def test_ifrnet_utils():

    # prepare inputs
    inputs = torch.rand(1, 3, 64, 64)
    flow = torch.rand(1, 2, 64, 64)

    # test warp
    output = warp(inputs, flow)
    assert output.shape == inputs.shape

    # test resize
    output = resize(inputs, 2)
    assert output.shape == torch.Size([1, 3, 128, 128])
