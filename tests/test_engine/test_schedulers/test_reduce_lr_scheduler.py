# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine import MessageHub

from mmagic.engine.schedulers import ReduceLR


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
