# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.airnet.dgrn_modules import DGG, default_conv


def test_dgg():
    model = DGG(default_conv, 64, 3, 5)

    # test attributes
    assert model.__class__.__name__ == 'DGG'

    # prepare data
    inputs = torch.rand(1, 64, 64, 64)
    targets = torch.rand(1, 64, 64, 64)

    # test on cpu
    output = model(inputs, inputs)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()

        output = model(inputs, inputs)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape
