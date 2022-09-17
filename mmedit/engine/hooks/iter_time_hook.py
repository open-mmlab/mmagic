# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Optional, Sequence, Union

from mmengine.hooks import IterTimerHook
from mmengine.registry import HOOKS
from mmengine.structures import BaseDataElement

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class GenIterTimerHook(IterTimerHook):
    """GenIterTimerHooks inherits from `:class:~mmengine.hooks.IterTimerHook`
    and overwrites `:meth:self._after_iter`.

    This hooks should be used along with
    `:class:mmedit.engine.runners.loops.GenValLoop` and
    `:class:mmedit.engine.runners.loops.GenTestLoop`.
    """

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[dict,
                                            Sequence[BaseDataElement]]] = None,
                    mode: str = 'train') -> None:
        """Calculating time for an iteration and updating "time"
        ``HistoryBuffer`` of ``runner.message_hub``. If `mode` is 'train', we
        take `runner.max_iters` as the total iterations and calculate the rest
        time. If `mode` in `val` or `test`, we use
        `runner.val_loop.total_length` or `runner.test_loop.total_length` as
        total number of iterations. If you want to know how `total_length` is
        calculated, please refers to
        `:meth:mmedit.engine.runners.loops.GenValLoop.run` and
        `:meth:mmedit.engine.runners.loops.GenTestLoop.run`.

        Args:
            runner (Runner): The runner of the training validation and
                testing process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict or sequence, optional): Outputs from model. Defaults
                to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        # Update iteration time in `runner.message_hub`.
        message_hub = runner.message_hub
        message_hub.update_scalar(f'{mode}/time', time.time() - self.t)
        self.t = time.time()
        window_size = runner.log_processor.window_size
        # Calculate eta every `window_size` iterations. Since test and val
        # loop will not update runner.iter, use `every_n_innter_iters`to check
        # the interval.
        if self.every_n_inner_iters(batch_idx, window_size):
            iter_time = message_hub.get_scalar(f'{mode}/time').mean(
                window_size)
            if mode == 'train':
                self.time_sec_tot += iter_time * window_size
                # Calculate average iterative time.
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                # Calculate eta.
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                runner.message_hub.update_info('eta', eta_sec)
            else:
                if mode == 'val':
                    total_length = runner.val_loop.total_length
                else:
                    total_length = runner.test_loop.total_length

                eta_sec = iter_time * (total_length - batch_idx - 1)
                runner.message_hub.update_info('eta', eta_sec)
