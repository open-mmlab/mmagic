import pytest
import torch
import torch.nn as nn
from mmedit.core import build_optimizer, build_optimizers
from mmedit.core.optimizer.registry import TORCH_OPTIMIZERS


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.bn = nn.BatchNorm2d(8)
        self.gn = nn.GroupNorm(3, 8)

    def forward(self, x):
        return x


def test_build_optimizer():
    with pytest.raises(TypeError):
        # paramwise_options should be a dict
        optimizer_cfg = dict(paramwise_options=['error'])
        model = ExampleModel()
        build_optimizer(model, optimizer_cfg)

    with pytest.raises(ValueError):
        # weight_decay cannot be None since bias_decay_mult is set
        optimizer_cfg = dict(
            paramwise_options=dict(bias_decay_mult=1, norm_decay_mult=1),
            lr=0.0001,
            weight_decay=None)
        model = ExampleModel()
        build_optimizer(model, optimizer_cfg)

    base_lr = 0.0001
    base_wd = 0.0002
    momentum = 0.9

    # basic config with ExampleModel
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    param_groups = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    assert len(param_groups['params']) == 6
    assert torch.equal(param_groups['params'][0], param_dict['conv1.weight'])
    assert torch.equal(param_groups['params'][1], param_dict['conv1.bias'])
    assert torch.equal(param_groups['params'][2], param_dict['bn.weight'])
    assert torch.equal(param_groups['params'][3], param_dict['bn.bias'])
    assert torch.equal(param_groups['params'][4], param_dict['gn.weight'])
    assert torch.equal(param_groups['params'][5], param_dict['gn.bias'])

    # basic config with Parallel model
    model = torch.nn.DataParallel(ExampleModel())
    optimizer = build_optimizer(model, optimizer_cfg)
    param_dict = dict(model.named_parameters())
    param_groups = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    assert len(param_groups['params']) == 6
    assert torch.equal(param_groups['params'][0],
                       param_dict['module.conv1.weight'])
    assert torch.equal(param_groups['params'][1],
                       param_dict['module.conv1.bias'])
    assert torch.equal(param_groups['params'][2],
                       param_dict['module.bn.weight'])
    assert torch.equal(param_groups['params'][3], param_dict['module.bn.bias'])
    assert torch.equal(param_groups['params'][4],
                       param_dict['module.gn.weight'])
    assert torch.equal(param_groups['params'][5], param_dict['module.gn.bias'])

    # Empty paramwise_options with ExampleModel
    optimizer_cfg['paramwise_options'] = dict()
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == momentum
        assert param_group['lr'] == base_lr
        assert param_group['weight_decay'] == base_wd

    # Empty paramwise_options with ExampleModel and no grad
    for param in model.parameters():
        param.requires_grad = False
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == momentum
        assert param_group['lr'] == base_lr
        assert param_group['weight_decay'] == base_wd

    # paramwise_options with ExampleModel
    paramwise_options = dict(
        bias_lr_mult=0.9, bias_decay_mult=0.8, norm_decay_mult=0.7)
    optimizer_cfg['paramwise_options'] = paramwise_options
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == momentum
    assert param_groups[0]['lr'] == base_lr
    assert param_groups[0]['weight_decay'] == base_wd
    assert param_groups[1]['lr'] == base_lr * 0.9
    assert param_groups[1]['weight_decay'] == base_wd * 0.8
    assert param_groups[2]['lr'] == base_lr
    assert param_groups[2]['weight_decay'] == base_wd * 0.7
    assert param_groups[3]['lr'] == base_lr
    assert param_groups[3]['weight_decay'] == base_wd * 0.7
    assert param_groups[4]['lr'] == base_lr
    assert param_groups[4]['weight_decay'] == base_wd * 0.7
    assert param_groups[5]['lr'] == base_lr
    assert param_groups[5]['weight_decay'] == base_wd * 0.7

    # paramwise_options with ExampleModel and no grad
    for param in model.parameters():
        param.requires_grad = False
    optimizer = build_optimizer(model, optimizer_cfg)
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['momentum'] == momentum
    assert optimizer.defaults['weight_decay'] == base_wd
    for i, (name, param) in enumerate(model.named_parameters()):
        param_group = param_groups[i]
        assert param_group['params'] == [param]
        assert param_group['momentum'] == momentum
        assert param_group['lr'] == base_lr
        assert param_group['weight_decay'] == base_wd


def test_torch_optimizers():
    torch_optimizers = [
        'ASGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'LBFGS', 'RMSprop',
        'Rprop', 'SGD', 'SparseAdam'
    ]
    assert set(torch_optimizers).issubset(set(TORCH_OPTIMIZERS))


class ExampleModel2(nn.Module):

    def __init__(self):
        super(ExampleModel2, self).__init__()
        self.model1 = nn.Conv2d(3, 8, kernel_size=3)
        self.model2 = nn.Conv2d(3, 4, kernel_size=3)

    def forward(self, x):
        return x


def test_build_optimizers():
    base_lr = 0.0001
    base_wd = 0.0002
    momentum = 0.9

    # basic config with ExampleModel2
    optimizer_cfg = dict(
        model1=dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum),
        model2=dict(
            type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum))
    model = ExampleModel2()
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
    model = torch.nn.DataParallel(ExampleModel2())
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

    # basic config with ExampleModel2 (one optimizer)
    optimizer_cfg = dict(
        type='SGD', lr=base_lr, weight_decay=base_wd, momentum=momentum)
    model = ExampleModel2()
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
    model = torch.nn.DataParallel(ExampleModel2())
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
