# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmedit.core import build_optimizers


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model1 = nn.Conv2d(3, 8, kernel_size=3)
        self.model2 = nn.Conv2d(3, 4, kernel_size=3)

    def forward(self, x):
        return x


def test_build_optimizers():
    base_lr = 0.0001
    base_wd = 0.0002
    momentum = 0.9

    # basic config with ExampleModel
    optimizer_cfg = dict(
        model1=dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum),
        model2=dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum))
    model = ExampleModel()
    optimizers = build_optimizers(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    assert isinstance(optimizers, dict)
    for i in range(2):
        optimizer = optimizers[f'model{i+1}']
        param_groups = optimizer.param_groups[0]
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == base_lr
        assert optimizer.defaults['momentum'] == momentum
        assert optimizer.defaults['weight_decay'] == base_wd
        assert len(param_groups['params']) == 2
        assert torch.equal(param_groups['params'][0],
                           param_dict[f'model{i+1}.weight'])
        assert torch.equal(param_groups['params'][1],
                           param_dict[f'model{i+1}.bias'])

    # basic config with Parallel model
    model = torch.nn.DataParallel(ExampleModel())
    optimizers = build_optimizers(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    assert isinstance(optimizers, dict)
    for i in range(2):
        optimizer = optimizers[f'model{i+1}']
        param_groups = optimizer.param_groups[0]
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == base_lr
        assert optimizer.defaults['momentum'] == momentum
        assert optimizer.defaults['weight_decay'] == base_wd
        assert len(param_groups['params']) == 2
        assert torch.equal(param_groups['params'][0],
                           param_dict[f'module.model{i+1}.weight'])
        assert torch.equal(param_groups['params'][1],
                           param_dict[f'module.model{i+1}.bias'])

    # basic config with ExampleModel (one optimizer)
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    model = ExampleModel()
    optimizer = build_optimizers(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    assert isinstance(optimizers, dict)
    param_groups = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    assert len(param_groups['params']) == 4
    assert torch.equal(param_groups['params'][0], param_dict['model1.weight'])
    assert torch.equal(param_groups['params'][1], param_dict['model1.bias'])
    assert torch.equal(param_groups['params'][2], param_dict['model2.weight'])
    assert torch.equal(param_groups['params'][3], param_dict['model2.bias'])

    # basic config with Parallel model (one optimizer)
    model = torch.nn.DataParallel(ExampleModel())
    optimizer = build_optimizers(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    assert isinstance(optimizers, dict)
    param_groups = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    assert len(param_groups['params']) == 4
    assert torch.equal(param_groups['params'][0],
                       param_dict['module.model1.weight'])
    assert torch.equal(param_groups['params'][1],
                       param_dict['module.model1.bias'])
    assert torch.equal(param_groups['params'][2],
                       param_dict['module.model2.weight'])
    assert torch.equal(param_groups['params'][3],
                       param_dict['module.model2.bias'])
