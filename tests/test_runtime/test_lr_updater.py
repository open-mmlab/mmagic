# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.core.scheduler.lr_updater import (LinearLrUpdaterHook,
                                              ReduceLrUpdaterHook)


class FakeLogBuffer(object):

    def __init__(self) -> None:
        self.output = {'PSNR': 30}


class FakeRunner(object):

    def __init__(self) -> None:
        self.epoch = 1
        self.max_epochs = 10
        self.iter = 1
        self.max_iters = 100
        self.optimizer = 1
        self.outputs = dict(log_vars=dict(loss=1))
        self.eval_result = {'PSNR': 30}
        self.log_buffer = FakeLogBuffer()


def test_linear_lr_updater_hook():
    fake_runner = FakeRunner()

    lr_updater = LinearLrUpdaterHook(by_epoch=False)
    lr = lr_updater.get_lr(fake_runner, 1)
    assert lr == 0.99

    lr_updater = LinearLrUpdaterHook(by_epoch=True)
    lr = lr_updater.get_lr(fake_runner, 1)
    assert lr == 0.9
    lr_updater.start = 10
    lr = lr_updater.get_lr(fake_runner, 1)
    assert lr == 1


def test_reduce_lr_updater_hook():
    fake_runner = FakeRunner()

    lr_updater = ReduceLrUpdaterHook(
        by_epoch=False, epoch_base_valid=False, verbose=True)
    lr_updater.get_regular_lr(fake_runner)
    lr_updater.num_bad_epochs = 1000
    fake_runner.optimizer = dict(a=1)
    lr_updater.regular_lr = dict(a=[0.1])
    lr_updater.num_bad_epochs = 1000
    lr_updater.factor = 0.1
    result = lr_updater.get_regular_lr(fake_runner)
    assert result['a'][0] - 0.01 < 1e-8
    lr_updater.eps = 1e8
    lr_updater.num_bad_epochs = 1000
    result = lr_updater.get_regular_lr(fake_runner)
    assert result['a'][0] - 0.01 < 1e-8

    lr_updater = ReduceLrUpdaterHook(
        by_epoch=False, val_metric='PSNR', mode='max', threshold_mode='abs')
    assert lr_updater.best == float('-inf')
    lr_updater.after_val_iter(fake_runner)
    assert lr_updater.best == 30
    assert lr_updater.num_bad_epochs == 0
    lr_updater.after_val_epoch(fake_runner)
    lr_updater.after_train_iter(fake_runner)
    lr_updater.after_train_epoch(fake_runner)
    lr_updater.cooldown_counter = 1
    lr_updater.after_val_iter(fake_runner)
    assert lr_updater.cooldown_counter == 0
    assert lr_updater.best == 30
    assert lr_updater.num_bad_epochs == 0
    lr_updater.after_val_iter(fake_runner)
    assert lr_updater.best == 30
    assert lr_updater.num_bad_epochs == 1
    lr_updater.warmup = 1
    lr_updater.warmup_by_epoch = False
    lr_updater.warmup_iters = 1000
    lr_updater.warmup_epochs = 1000
    lr_updater.after_val_iter(fake_runner)
    assert lr_updater.best == 30
    assert lr_updater.num_bad_epochs == 1

    lr_updater = ReduceLrUpdaterHook(
        by_epoch=True, val_metric='PSNR', mode='max', threshold_mode='rel')
    lr_updater.after_val_iter(fake_runner)
    lr_updater.after_val_epoch(fake_runner)
    assert lr_updater.best == 30
    assert lr_updater.num_bad_epochs == 0
    lr_updater.after_train_iter(fake_runner)
    lr_updater.after_train_epoch(fake_runner)
    lr_updater.cooldown_counter = 1
    lr_updater.after_val_epoch(fake_runner)
    assert lr_updater.cooldown_counter == 0
    lr_updater.after_val_epoch(fake_runner)
    assert lr_updater.best == 30
    assert lr_updater.num_bad_epochs == 1
    lr_updater.warmup = 1
    lr_updater.warmup_by_epoch = True
    lr_updater.warmup_epochs = 1000
    lr_updater.after_val_epoch(fake_runner)
    assert lr_updater.best == 30
    assert lr_updater.num_bad_epochs == 1

    lr_updater = ReduceLrUpdaterHook(
        by_epoch=False, mode='min', threshold_mode='abs')
    assert lr_updater.best == float('inf')
    lr_updater.after_val_iter(fake_runner)
    lr_updater.after_val_epoch(fake_runner)
    lr_updater.after_train_iter(fake_runner)
    lr_updater.after_train_epoch(fake_runner)
    assert lr_updater.best == 1
    assert lr_updater.num_bad_epochs == 0
    lr_updater.cooldown_counter = 1
    lr_updater.after_train_iter(fake_runner)
    assert lr_updater.cooldown_counter == 0
    lr_updater.after_train_iter(fake_runner)
    assert lr_updater.best == 1
    assert lr_updater.num_bad_epochs == 1
    lr_updater.warmup = 1
    lr_updater.warmup_by_epoch = False
    lr_updater.warmup_iters = 1000
    lr_updater.warmup_epochs = 1000
    lr_updater.after_train_iter(fake_runner)
    assert lr_updater.best == 1
    assert lr_updater.num_bad_epochs == 1

    lr_updater = ReduceLrUpdaterHook(
        by_epoch=True, mode='min', threshold_mode='rel')
    lr_updater.after_val_iter(fake_runner)
    lr_updater.after_val_epoch(fake_runner)
    lr_updater.after_train_iter(fake_runner)
    lr_updater.after_train_epoch(fake_runner)
    assert lr_updater.best == 1
    assert lr_updater.num_bad_epochs == 0
    lr_updater.cooldown_counter = 1
    lr_updater.after_train_epoch(fake_runner)
    assert lr_updater.cooldown_counter == 0
    lr_updater.after_train_epoch(fake_runner)
    assert lr_updater.best == 1
    assert lr_updater.num_bad_epochs == 1
    lr_updater.warmup = 1
    lr_updater.warmup_by_epoch = True
    lr_updater.warmup_epochs = 1000
    lr_updater.after_train_epoch(fake_runner)
    assert lr_updater.best == 1
    assert lr_updater.num_bad_epochs == 1
