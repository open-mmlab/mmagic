# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models import build_backbone


def test_tof_vfi_net():

    model_cfg = dict(
        type='TOFlowVFINet', norm_cfg=None, load_pretrained_spynet=None)

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'TOFlowVFINet'

    # prepare data
    inputs = torch.rand(1, 2, 3, 256, 248)
    target = torch.rand(1, 3, 256, 248)

    # test on cpu
    output = model(inputs)
    assert torch.is_tensor(output)
    assert output.shape == target.shape
    with pytest.raises(OSError):
        model.init_weights('')
    with pytest.raises(TypeError):
        model.init_weights(1)

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        output = model(inputs)
        output = model(inputs, True)
        assert torch.is_tensor(output)
        assert output.shape == target.shape
