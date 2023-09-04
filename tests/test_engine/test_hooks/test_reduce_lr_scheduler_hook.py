# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from mmengine import MessageHub

from mmagic.engine.hooks import ReduceLRSchedulerHook


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


def test_reduce_lr_scheduler_hook():

    scheduler = [Mock()]
    scheduler[0].step = Mock()

    runner = Mock()
    runner.epoch = 0
    runner.iter = 0
    runner.param_schedulers = scheduler
    runner.message_hub = MessageHub.get_instance('test-reduce-lr-scheduler')
    runner.message_hub.update_scalar('train/loss', 0.1)

    hook = ReduceLRSchedulerHook(val_metric=None, by_epoch=True, interval=2)
    scheduler[0].by_epoch = True
    hook.after_train_iter(runner, 0, data_batch=[dict(a=1)] * 3)
    runner.message_hub.update_scalar('train/loss', 0.2)
    hook.after_train_iter(runner, 0, data_batch=[dict(a=1)] * 2)
    assert abs(hook.sum_value - 0.7) < 1e-8
    assert hook.count == 5
    hook.after_train_epoch(runner)
    runner.epoch += 1
    assert abs(hook.sum_value - 0.7) < 1e-8
    assert hook.count == 5
    hook.after_train_epoch(runner)
    scheduler[0].step.assert_called()
    assert abs(hook.sum_value - 0) < 1e-8
    assert hook.count == 0
    value = hook.message_hub.get_scalar('value').current()
    assert abs(value - 0.14) < 1e-8
    hook.after_val_iter(runner, 0, data_batch=[dict(a=1)] * 2)
    hook.after_val_epoch(runner)

    hook = ReduceLRSchedulerHook(val_metric=None, by_epoch=False, interval=2)
    scheduler[0].by_epoch = False
    runner.message_hub.update_scalar('train/loss', 0.1)
    hook.after_train_iter(runner, 0, data_batch=[dict(a=1)] * 3)
    runner.iter += 1
    assert abs(hook.sum_value - 0.3) < 1e-8
    assert hook.count == 3
    runner.message_hub.update_scalar('train/loss', 0.3)
    hook.after_train_iter(runner, 0, data_batch=[dict(a=1)] * 2)
    scheduler[0].step.assert_called()
    value = hook.message_hub.get_scalar('value').current()
    assert abs(value - 0.18) < 1e-8
    assert abs(hook.sum_value - 0) < 1e-8
    assert hook.count == 0
    hook.after_train_epoch(runner)
    hook.after_val_iter(runner, 0, data_batch=[dict(a=1)] * 2)
    hook.after_val_epoch(runner)

    hook = ReduceLRSchedulerHook(val_metric='PSNR', by_epoch=True, interval=2)
    scheduler[0].by_epoch = False
    hook.after_train_iter(runner, 0)
    hook.after_train_epoch(runner)
    runner.epoch = 0
    hook.after_val_epoch(runner, metrics=dict(PSNR=40))
    assert abs(hook.sum_value - 40) < 1e-8, hook.sum_value
    assert hook.count == 1
    runner.epoch = 1
    hook.after_val_epoch(runner, metrics=dict(PSNR=50))
    scheduler[0].step.assert_called()
    value = hook.message_hub.get_scalar('value').current()
    assert abs(value - 45) < 1e-8
    assert abs(hook.sum_value - 0) < 1e-8, hook.sum_value
    assert hook.count == 0

    hook = ReduceLRSchedulerHook(val_metric='PSNR', by_epoch=False, interval=2)
    scheduler[0].by_epoch = False
    hook.after_train_iter(runner, 0)
    hook.after_train_epoch(runner)
    runner.epoch = 0
    hook.after_val_epoch(runner, metrics=dict(PSNR=40))
    scheduler[0].step.assert_called()
    value = hook.message_hub.get_scalar('value').current()
    assert abs(value - 40) < 1e-8
    assert abs(hook.sum_value - 0) < 1e-8, hook.sum_value
    assert hook.count == 0
    runner.epoch = 1
    runner.param_schedulers = dict(scheduler=scheduler)
    hook.after_val_epoch(runner, metrics=dict(PSNR=50))
    scheduler[0].step.assert_called()
    value = hook.message_hub.get_scalar('value').current()
    assert abs(value - 50) < 1e-8
    assert abs(hook.sum_value - 0) < 1e-8, hook.sum_value
    assert hook.count == 0

    with pytest.raises(AssertionError):
        runner.param_schedulers = dict(a='')
        hook.after_val_epoch(runner, metrics=dict(PSNR=50))

    with pytest.raises(TypeError):
        runner.param_schedulers = ''
        hook.after_val_epoch(runner, metrics=dict(PSNR=50))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
