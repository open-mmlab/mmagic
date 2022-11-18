# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

from mmengine import MessageHub
from mmengine.hooks import ParamSchedulerHook
from mmengine.runner import Runner

from mmedit.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class ReduceLRSchedulerHook(ParamSchedulerHook):
    """A hook to update learning rate.

    Args:
        val_metric (str): The metric of validation. If val_metric is not None,
            we check val_metric to reduce learning. Default: None.
        by_epoch (bool): Whether to update by epoch. Default: True.
        interval (int): The interval of iterations to update. Default: 1.
    """

    def __init__(self,
                 val_metric: str = None,
                 by_epoch=True,
                 interval=1) -> None:
        super().__init__()

        self.message_hub = MessageHub.get_instance('reduce_lr')

        self.val_metric = val_metric
        self.by_epoch = by_epoch
        self.interval = interval
        self.sum_value = 0
        self.count = 0

    def _calculate_average_value(self):
        value = self.sum_value / self.count
        self.sum_value = 0
        self.count = 0
        self.message_hub.update_scalar('value', value)

    def after_train_epoch(self, runner: Runner):
        """Call step function for each scheduler after each train epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        # If val_metric is not None, we check val_metric to reduce learning
        if self.val_metric is not None:
            return

        if self.every_n_epochs(runner, self.interval):
            self._calculate_average_value()
            super().after_train_epoch(runner=runner)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Call step function for each scheduler after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                In order to keep this interface consistent with other hooks,
                we keep ``data_batch`` here. Defaults to None.
            outputs (dict, optional): Outputs from model.
                In order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here. Defaults to None.
        """

        # If val_metric is not None, we check val_metric to reduce learning
        if self.val_metric is not None:
            return

        current = runner.message_hub.get_scalar('train/loss').current()
        self.sum_value += current * len(data_batch)
        self.count += len(data_batch)

        if self.by_epoch:
            return

        if self.every_n_train_iters(runner, self.interval):
            self._calculate_average_value()
            super().after_train_iter(
                runner=runner,
                batch_idx=batch_idx,
                data_batch=data_batch,
                outputs=outputs)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None):
        """Call step function for each scheduler after each validation epoch.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict, optional): The metrics of validation. Default: None.
        """
        # If val_metric is None, we check training loss to reduce learning
        # rate.
        if self.val_metric is None:
            return

        if self.val_metric not in metrics:
            raise KeyError(f'{self.val_metric} is not found in metrics')

        self.sum_value += metrics[self.val_metric]
        self.count += 1

        if not self.by_epoch or self.every_n_epochs(runner, self.interval):
            # if self.by_epoch is False,
            # call val after several iter
            # and update LR in each ``after_val_epoch``
            self._calculate_average_value()

            def step(param_schedulers):
                assert isinstance(param_schedulers, list)
                for scheduler in param_schedulers:
                    scheduler.step()

            if isinstance(runner.param_schedulers, list):
                step(runner.param_schedulers)
            elif isinstance(runner.param_schedulers, dict):
                for param_schedulers in runner.param_schedulers.values():
                    step(param_schedulers)
            else:
                raise TypeError(
                    'runner.param_schedulers should be list of ParamScheduler '
                    'or a dict containing list of ParamScheduler, '
                    f'but got {runner.param_schedulers}')
