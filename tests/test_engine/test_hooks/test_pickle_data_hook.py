# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from mmagic.engine.hooks import PickleDataHook


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.randn(10, 10)
        self.b = np.random.random((10, 10))


def test_PickleDataHook():
    hook = PickleDataHook(
        output_dir='./', data_name_list=['a', 'b', 'c'], interval=3)
    runner = MagicMock()
    runner.work_dir = './test/data'
    runner.iter = 0
    runner.model = ToyModel()

    # test after train iter
    hook.after_train_iter(runner, 0, None, None)
    hook.data_name_list = ['a', 'b']
    hook.after_train_iter(runner, 0, None, None)

    runner.model = MagicMock()
    runner.model.module = ToyModel()
    hook._pickle_data(runner)

    runner.iter = 2
    hook.after_train_iter(runner, 0, None, None)

    # test after run
    hook.after_run(runner)

    # test before run
    hook.before_run(runner)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
