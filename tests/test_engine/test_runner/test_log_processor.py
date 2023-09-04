# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

from mmagic.engine import LogProcessor as LogProcessor


class TestLogProcessor:

    def setup(self):
        runner = MagicMock()
        runner.epoch = 1
        runner.iter = 10
        runner.max_epochs = 10
        runner.max_iters = 50
        runner.train_dataloader = [0] * 20
        runner.val_dataloader = [0] * 10
        runner.test_dataloader = [0] * 5
        runner.train_loop.dataloader = [0] * 20
        runner.val_loop.total_length = 10
        runner.test_loop.total_length = 5
        self.runner = runner

    def test_get_dataloader_size(self):
        log_processor = LogProcessor(by_epoch=True)
        del self.runner.train_loop.total_length
        assert log_processor._get_dataloader_size(self.runner, 'train') == 20
        assert log_processor._get_dataloader_size(self.runner, 'val') == 10
        assert log_processor._get_dataloader_size(self.runner, 'test') == 5


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
