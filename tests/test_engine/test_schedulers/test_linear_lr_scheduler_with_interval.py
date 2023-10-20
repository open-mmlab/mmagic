# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine import MessageHub

from mmagic.engine.schedulers import LinearLrInterval


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestLinearLrInterval(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.model = ToyModel()
        lr = 1
        self.optimizer = optim.SGD([{
            'params': self.model.conv1.parameters()
        }],
                                   lr=lr,
                                   momentum=0.01,
                                   weight_decay=5e-4)

    def test_step(self):
        targets = torch.linspace(1., 0., 11)
        param_scheduler = LinearLrInterval(
            self.optimizer,
            interval=1,
            by_epoch=False,
            start_factor=1.0,
            end_factor=0,
            begin=0,
            end=10)
        messageHub = MessageHub.get_current_instance()
        for step in range(10):
            messageHub.update_info('iter', step)
            param_scheduler.step()
            np.testing.assert_almost_equal(param_scheduler._get_value(),
                                           targets[step].item())


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
