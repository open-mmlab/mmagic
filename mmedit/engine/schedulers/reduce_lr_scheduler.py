# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import MessageHub
from mmengine.optim import _ParamScheduler

from mmedit.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class ReduceLR(_ParamScheduler):
    """Decays the learning rate of each parameter group by linearly changing
    small multiplicative factor until the number of epoch reaches a pre-defined
    milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    learning rate from outside this scheduler.

    Note:
        The learning rate of each parameter group will be update at regular
            intervals.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        mode (str, optional): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float, optional): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int, optional): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float, optional): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str, optional): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int, optional): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value.
            Default: 0.
        eps (float, optional): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
    """

    def __init__(self,
                 optimizer,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: float = 0.,
                 eps: float = 1e-8,
                 **kwargs):

        super().__init__(optimizer=optimizer, param_name='lr', **kwargs)

        self.message_hub = MessageHub.get_instance('reduce_lr')

        if mode not in ['min', 'max']:
            raise ValueError(
                'mode must be one of "min" or "max", instead got {mode}')
        self.mode = mode

        if factor >= 1.0 or factor < 0:
            raise ValueError('Factor should be < 1.0 and >=0')
        self.factor = factor

        self.patience = patience
        self.threshold = threshold

        if threshold_mode not in ['rel', 'abs']:
            raise ValueError('thresh_mode must be one of "rel" or "abs",'
                             f'instead got {threshold_mode}')
        self.threshold_mode = threshold_mode

        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.min_lr = min_lr
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(self.mode)
        self._reset()

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""

        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        current = self.message_hub.get_scalar('value').current()
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            results = []
            for group in self.optimizer.param_groups:
                regular_lr = group[self.param_name]
                if regular_lr - regular_lr * self.factor > self.eps:
                    regular_lr = max(regular_lr * self.factor, self.min_lr)
                results.append(regular_lr)
            return results

        else:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

    def _init_is_better(self, mode):
        if mode == 'min':
            self.mode_worse = float('inf')
        else:
            self.mode_worse = float('-inf')

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = 1. + self.threshold
            return a > best * rel_epsilon
        else:
            return a > best + self.threshold

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
