# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import pytest
import torch
import torch.nn as nn

from mmagic.models.utils import (build_module, generation_init_weights,
                                 get_module_device, get_valid_num_batches,
                                 set_requires_grad)
from mmagic.registry import MODELS


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

    # test update init info
    module = nn.BatchNorm2d(3)
    module._params_init_info = defaultdict(dict)
    for name, param in module.named_parameters():
        module._params_init_info[param][
            'init_info'] = f'The value is the same before and ' \
                            f'after calling `init_weights` ' \
                            f'of {module.__class__.__name__} '
        module._params_init_info[param]['tmp_mean_value'] = param.data.mean(
        ).cpu()
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


def test_get_valid_num_batches():

    # 1. batch inputs is Tensor
    assert get_valid_num_batches(torch.rand(5, 3, 4, 4)) == 5

    # 2. batch inputs is dict
    batch_inputs = dict(num_batches=1, img=torch.randn(1, 3, 5, 5))
    assert get_valid_num_batches(batch_inputs) == 1

    # 3. batch inputs is dict but no tensor in it
    batch_inputs = dict(aaa='aaa', bbb='bbb')
    assert get_valid_num_batches(batch_inputs) == 1

    # 4. test no batch input and no data samples
    assert get_valid_num_batches() == 1

    # 5. test batch is None but data sample is not None
    data_samples = [None, None]
    assert get_valid_num_batches(None, data_samples) == 2

    # 6. test batch input and data sample are neither not None
    batch_inputs = dict(num_batches=2, img=torch.randn(2, 3, 5, 5))
    data_samples = [None, None]
    assert get_valid_num_batches(batch_inputs, data_samples) == 2


def test_build_module():
    module = nn.Conv2d(3, 3, 3)
    assert build_module(module, MODELS) == module

    @MODELS.register_module()
    class MiniModule(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            for k, v in kwargs.items():
                setattr(self, k, v)

        def forward(self, *args, **kwargs):
            return

    module_cfg = dict(type='MiniModule', attr1=1, attr2=2)
    module = build_module(module_cfg, MODELS)
    assert module.attr1 == 1
    assert module.attr2 == 2

    with pytest.raises(TypeError):
        build_module('MiniModule', MODELS)

    # remove the registered module
    MODELS._module_dict.pop('MiniModule')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
