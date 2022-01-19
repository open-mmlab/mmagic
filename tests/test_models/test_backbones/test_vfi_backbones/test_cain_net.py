# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models import build_backbone


def test_cain_net():

    model_cfg = dict(type='CAINNet')

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'CAINNet'

    # prepare data
    inputs0 = torch.rand(1, 2, 3, 5, 5)
    target0 = torch.rand(1, 3, 5, 5)
    inputs = torch.rand(1, 2, 3, 256, 248)
    target = torch.rand(1, 3, 256, 248)

    # test on cpu
    output = model(inputs)
    output = model(inputs, padding_flag=True)
    model(inputs0, padding_flag=True)
    assert torch.is_tensor(output)
    assert output.shape == target.shape
    with pytest.raises(AssertionError):
        output = model(inputs[:, :1])
    with pytest.raises(OSError):
        model.init_weights('')
    with pytest.raises(TypeError):
        model.init_weights(1)

    model_cfg = dict(type='CAINNet', norm='in')
    model = build_backbone(model_cfg)
    model(inputs)
    model_cfg = dict(type='CAINNet', norm='bn')
    model = build_backbone(model_cfg)
    model(inputs)
    with pytest.raises(ValueError):
        model_cfg = dict(type='CAINNet', norm='lys')
        build_backbone(model_cfg)

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        output = model(inputs)
        output = model(inputs, True)
        assert torch.is_tensor(output)
        assert output.shape == target.shape
        inputs0 = inputs0.cuda()
        target0 = target0.cuda()
        model(inputs0, padding_flag=True)

        model_cfg = dict(type='CAINNet', norm='in')
        model = build_backbone(model_cfg).cuda()
        model(inputs)
        model_cfg = dict(type='CAINNet', norm='bn')
        model = build_backbone(model_cfg).cuda()
        model(inputs)
        with pytest.raises(ValueError):
            model_cfg = dict(type='CAINNet', norm='lys')
            build_backbone(model_cfg).cuda()
