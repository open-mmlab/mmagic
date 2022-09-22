# Copyright (c) OpenMMLab. All rights reserved.
import copy
import datetime
from typing import Tuple

import torch
from mmengine.registry import LOG_PROCESSORS
from mmengine.runner import LogProcessor


@LOG_PROCESSORS.register_module()  # type: ignore
class GenLogProcessor(LogProcessor):
    """GenLogProcessor inherits from `:class:~mmengine.logging.LogProcessor`
    and overwrites `:meth:self.get_log_after_iter`.

    This log processor should be used along with
    `:class:mmedit.engine.runners.loops.GenValLoop` and
    `:class:mmedit.engine.runners.loops.GenTestLoop`.
    """

    def get_log_after_iter(self, runner, batch_idx: int,
                           mode: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing epoch.

        If `mode` is in 'val' or 'test', we use `runner.val_loop.total_length`
        and `runner.test_loop.total_length` as the total number of iterations
        shown in log. If you want to know how `total_length` is calculated,
        please refers to `:meth:mmedit.engine.runners.loops.GenValLoop.run` and
        `:meth:mmedit.engien.runners.loops.GenTestLoop.run`.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.
        Return:
            Tuple(dict, str): Formatted log dict/string which will be
                recorded by :obj:`runner.message_hub` and
                :obj:`runner.visualizer`.
        """
        assert mode in ['train', 'test', 'val']
        if mode == 'train':
            return super().get_log_after_iter(runner, batch_idx, 'train')

        # use our own defined method in test and val mode

        current_loop = self._get_cur_loop(runner, mode)
        cur_iter = self._get_iter(runner, batch_idx=batch_idx)
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        custom_cfg_copy = self._parse_windows_size(runner, batch_idx)
        # tag is used to write log information to different backends.
        tag = self._collect_scalars(custom_cfg_copy, runner, mode)
        # `log_tag` will pop 'lr' and loop other keys to `log_str`.
        log_tag = copy.deepcopy(tag)
        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.startswith('lr'):
                log_tag.pop(key)
                lr_str_list.append(f'{key}: {value:.{self.num_digits}e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by `total_length`)

        total_length = current_loop.total_length

        if self.by_epoch:
            if mode == 'val':
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (f'Epoch({mode}) [{cur_epoch}]'
                           f'[{cur_iter}/{total_length}]  ')
            else:
                log_str = (f'Epoch({mode}) ' f'[{cur_iter}/{total_length}]  ')
        else:
            log_str = (f'Iter({mode}) [{batch_idx+1}/{total_length}]  ')
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}  '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f'eta: {eta_str}  '
            log_str += (f'time: {tag["time"]:.{self.num_digits}f}  '
                        f'data_time: {tag["data_time"]:.{self.num_digits}f}  ')
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda is available, the max memory occupied should be calculated.
        if torch.cuda.is_available():
            log_str += f'memory: {self._get_max_memory(runner)}  '
        # Loop left keys to fill `log_str`.
        if mode in ('train', 'val'):
            log_items = []
            for name, val in log_tag.items():
                if mode == 'val' and not name.startswith('val/loss'):
                    continue
                if isinstance(val, float):
                    val = f'{val:.{self.num_digits}f}'
                log_items.append(f'{name}: {val}')
            log_str += '  '.join(log_items)
        return tag, log_str

    def get_log_after_epoch(self, runner, batch_idx: int,
                            mode: str) -> Tuple[dict, str]:
        """Format log string after validation or testing epoch.

        We use `runner.val_loop.total_length` and
        `runner.test_loop.total_length` as the total number of iterations
        shown in log. If you want to know how `total_length` is calculated,
        please refers to `:meth:mmgen.core.runners.loops.GenValLoop.run` and
        `:meth:mmgen.core.runners.loops.GenTestLoop.run`.

        Args:
            runner (Runner): The runner of validation/testing phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner.

        Return:
            Tuple(dict, str): Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in [
            'test', 'val'
        ], ('`_get_metric_log_str` only accept val or test mode, but got '
            f'{mode}')
        cur_loop = self._get_cur_loop(runner, mode)
        total_length = cur_loop.total_length

        custom_cfg_copy = self._parse_windows_size(runner, batch_idx)
        # tag is used to write log information to different backends.
        tag = self._collect_scalars(custom_cfg_copy, runner, mode)
        # By epoch:
        #     Epoch(val) [10][1000/1000]  ...
        #     Epoch(test) [1000/1000] ...
        # By iteration:
        #     Iteration(val) [1000/1000]  ...
        #     Iteration(test) [1000/1000]  ...
        if self.by_epoch:
            if mode == 'val':
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (f'Epoch({mode}) [{cur_epoch}][{total_length}/'
                           f'{total_length}]  ')
            else:
                log_str = (f'Epoch({mode}) [{total_length}/{total_length}]  ')

        else:
            log_str = (f'Iter({mode}) [{total_length}/{total_length}]  ')
        # `time` and `data_time` will not be recorded in after epoch log
        # message.
        log_items = []
        for name, val in tag.items():
            if name in ('time', 'data_time'):
                continue
            if isinstance(val, float):
                val = f'{val:.{self.num_digits}f}'
            log_items.append(f'{name}: {val}')
        log_str += '  '.join(log_items)
        return tag, log_str
