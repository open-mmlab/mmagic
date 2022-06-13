# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine import MessageHub
from mmengine.optim import _ParamScheduler
from mmengine.testing import assert_allclose

from mmedit.optimizer import LinearLRWithInterval, ReduceLR


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestLRScheduler(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.model = ToyModel()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.01, weight_decay=5e-4)

    def _test_scheduler_value(self,
                              schedulers,
                              targets,
                              epochs=10,
                              param_name='lr'):
        if isinstance(schedulers, _ParamScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            self.optimizer.step()
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                assert_allclose(
                    target[epoch],
                    param_group[param_name],
                    msg='{} is wrong in epoch {}: expected {}, got {}'.format(
                        param_name, epoch, target[epoch],
                        param_group[param_name]),
                    atol=1e-5,
                    rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def test_linear_scheduler(self):
        with self.assertRaises(ValueError):
            LinearLRWithInterval(self.optimizer, start_factor=10, end=900)
        with self.assertRaises(ValueError):
            LinearLRWithInterval(self.optimizer, start_factor=-1, end=900)
        with self.assertRaises(ValueError):
            LinearLRWithInterval(self.optimizer, end_factor=1.001, end=900)
        with self.assertRaises(ValueError):
            LinearLRWithInterval(self.optimizer, end_factor=-0.00001, end=900)
        # lr = 0.025     if epoch in [0, 1]
        # lr = 0.03125   if epoch in [2, 3]
        # lr = 0.0375    if epoch in [4, 5]
        # lr = 0.04375   if epoch in [6, 7]
        # lr = 0.005     if epoch >= 8
        epochs = 12
        start_factor = 1.0 / 2
        end_factor = 1.0
        iters = 8
        interpolation = []
        for i in range(iters // 2):
            interpolation.extend(
                [start_factor + i *
                 (end_factor - start_factor) / iters * 2] * 2)
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (
            epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]

        scheduler = LinearLRWithInterval(
            self.optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            end=iters + 1,
            interval=2)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_linear_scheduler_convert_iterbased(self):
        epochs = 10
        start_factor = 1.0
        end_factor = 0
        begin = 5
        end = 10
        epoch_length = 11
        interval = 5

        iters = end * epoch_length - 1 - begin * epoch_length
        interpolation = []
        for i in range(iters // interval):
            interpolation.extend([
                start_factor + i *
                (end_factor - start_factor) / iters * interval
            ] * interval)

        single_targets = [0.05 * start_factor] * (begin * epoch_length) + [
            x * 0.05 for x in interpolation
        ]
        single_targets += [0.05 * end_factor] * (
            epochs * epoch_length - len(single_targets))
        targets = [single_targets, [x * epochs for x in single_targets]]

        scheduler = LinearLRWithInterval.build_iter_from_epoch(
            self.optimizer,
            begin=begin,
            end=end,
            interval=interval,
            start_factor=start_factor,
            epoch_length=epoch_length)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_reduce_lr_scheduler(self):

        message_hub = MessageHub.get_instance('reduce_lr')

        scheduler = ReduceLR(self.optimizer, patience=1)
        scheduler.last_step = 0
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 0
        scheduler.last_step = 1
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 0
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 1
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert abs(results[0] - 0.005) < 1e-8
        assert scheduler.num_bad_epochs == 0

        scheduler = ReduceLR(self.optimizer, patience=1, mode='max')
        scheduler.last_step = 0
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 0
        scheduler.last_step = 1
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 0
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 1
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert abs(results[0] - 0.005) < 1e-8
        assert scheduler.num_bad_epochs == 0

        scheduler = ReduceLR(
            self.optimizer, patience=1, mode='max', threshold_mode='abs')
        scheduler.last_step = 1
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 0
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 1

        scheduler = ReduceLR(
            self.optimizer, patience=1, mode='min', threshold_mode='abs')
        scheduler.last_step = 1
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 0
        message_hub.update_scalar('value', 1)
        results = scheduler._get_value()
        assert results == [0.05]
        assert scheduler.num_bad_epochs == 1

        with pytest.raises(ValueError):
            ReduceLR(self.optimizer, mode='ysli')
        with pytest.raises(ValueError):
            ReduceLR(self.optimizer, threshold_mode='ysli')
        with pytest.raises(ValueError):
            ReduceLR(self.optimizer, factor=1.5)
