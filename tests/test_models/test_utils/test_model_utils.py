# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch
import torch.nn as nn

from mmedit.models.utils import (generation_init_weights, get_module_device,
                                 set_requires_grad)


def test_generation_init_weights():
    # Conv
    module = nn.Conv2d(3, 3, 1)
    module_tmp = copy.deepcopy(module)
    generation_init_weights(module, init_type='normal', init_gain=0.02)
    generation_init_weights(module, init_type='xavier', init_gain=0.02)
    generation_init_weights(module, init_type='kaiming')
    generation_init_weights(module, init_type='orthogonal', init_gain=0.02)
    with pytest.raises(NotImplementedError):
        generation_init_weights(module, init_type='abc')
    assert not torch.equal(module.weight.data, module_tmp.weight.data)

    # Linear
    module = nn.Linear(3, 1)
    module_tmp = copy.deepcopy(module)
    generation_init_weights(module, init_type='normal', init_gain=0.02)
    generation_init_weights(module, init_type='xavier', init_gain=0.02)
    generation_init_weights(module, init_type='kaiming')
    generation_init_weights(module, init_type='orthogonal', init_gain=0.02)
    with pytest.raises(NotImplementedError):
        generation_init_weights(module, init_type='abc')
    assert not torch.equal(module.weight.data, module_tmp.weight.data)

    # BatchNorm2d
    module = nn.BatchNorm2d(3)
    module_tmp = copy.deepcopy(module)
    generation_init_weights(module, init_type='normal', init_gain=0.02)
    assert not torch.equal(module.weight.data, module_tmp.weight.data)


def test_set_requires_grad():
    model = torch.nn.Conv2d(1, 3, 1, 1)
    set_requires_grad(model, False)
    for param in model.parameters():
        assert not param.requires_grad


def test_get_module_device_cpu():
    device = get_module_device(nn.Conv2d(3, 3, 3, 1, 1))
    assert device == torch.device('cpu')

    # The input module should contain parameters.
    with pytest.raises(ValueError):
        get_module_device(nn.Flatten())
