# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from mmengine.logging import MessageHub

from mmagic.engine import IterTimerHook


def time_patch():
    if not hasattr(time_patch, 'time'):
        time_patch.time = 0
    else:
        time_patch.time += 1
    return time_patch.time


class TestIterTimerHook(TestCase):

    def setUp(self) -> None:
        self.hook = IterTimerHook()

    def test_init(self):
        assert self.hook.time_sec_tot == 0
        assert self.hook.start_iter == 0

    def test_before_train(self):
        runner = MagicMock()
        runner.iter = 1
        self.hook.before_train(runner)
        assert self.hook.start_iter == 1

    def test_before_epoch(self):
        runner = Mock()
        self.hook._before_epoch(runner)
        assert isinstance(self.hook.t, float)

    @patch('time.time', MagicMock(return_value=1))
    def test_before_iter(self):
        runner = MagicMock()
        runner.log_buffer = dict()
        self.hook._before_epoch(runner)
        for mode in ('train', 'val', 'test'):
            self.hook._before_iter(runner, batch_idx=1, mode=mode)
            runner.message_hub.update_scalar.assert_called_with(
                f'{mode}/data_time', 0)

    @patch('time.time', time_patch)
    def test_after_iter(self):
        runner = MagicMock()
        runner.log_buffer = dict()
        runner.log_processor.window_size = 10
        runner.max_iters = 100
        runner.iter = 0
        runner.test_loop.total_length = 20
        runner.val_loop.total_length = 20
        self.hook._before_epoch(runner)
        self.hook.before_run(runner)
        self.hook._after_iter(runner, batch_idx=1)
        runner.message_hub.update_scalar.assert_called()
        runner.message_hub.get_log.assert_not_called()
        runner.message_hub.update_info.assert_not_called()
        runner.message_hub = MessageHub.get_instance('test_iter_timer_hook')
        runner.iter = 9
        # eta = (100 - 10) / 1
        self.hook._after_iter(runner, batch_idx=89)
        assert runner.message_hub.get_info('eta') == 90
        self.hook._after_iter(runner, batch_idx=9, mode='val')
        assert runner.message_hub.get_info('eta') == 10
        self.hook._after_iter(runner, batch_idx=19, mode='test')
        assert runner.message_hub.get_info('eta') == 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
