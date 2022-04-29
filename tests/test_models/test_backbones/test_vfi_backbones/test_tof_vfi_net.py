# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models import build_backbone


def test_tof_vfi_net():

    model_cfg = dict(type='TOFlowVFINet')

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'TOFlowVFINet'

    # prepare data
    inputs = torch.rand(1, 2, 3, 256, 256)

    # test on cpu
    output = model(inputs)
    assert torch.is_tensor(output)
    assert output.shape == (1, 3, 256, 256)

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        output = model(inputs)
        assert torch.is_tensor(output)
        assert output.shape == (1, 3, 256, 256)

    with pytest.raises(OSError):
        model.init_weights('')
    with pytest.raises(TypeError):
        model.init_weights(1)
    with pytest.raises(OSError):
        model_cfg = dict(
            type='TOFlowVFINet', flow_cfg=dict(norm_cfg=None, pretrained=''))
        model = build_backbone(model_cfg)
    with pytest.raises(TypeError):
        model_cfg = dict(
            type='TOFlowVFINet', flow_cfg=dict(norm_cfg=None, pretrained=1))
        model = build_backbone(model_cfg)
