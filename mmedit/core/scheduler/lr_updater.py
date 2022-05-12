# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class LinearLrUpdaterHook(LrUpdaterHook):
    """Linear learning rate scheduler for image generation.

    In the beginning, the learning rate is 'base_lr' defined in mmcv.
    We give a target learning rate 'target_lr' and a start point 'start'
    (iteration / epoch). Before 'start', we fix learning rate as 'base_lr';
    After 'start', we linearly update learning rate to 'target_lr'.

    Args:
        target_lr (float): The target learning rate. Default: 0.
        start (int): The start point (iteration / epoch, specified by args
            'by_epoch' in its parent class in mmcv) to update learning rate.
            Default: 0.
        interval (int): The interval to update the learning rate. Default: 1.
    """

    def __init__(self, target_lr=0, start=0, interval=1, **kwargs):
        super().__init__(**kwargs)
        self.target_lr = target_lr
        self.start = start
        self.interval = interval

    def get_lr(self, runner, base_lr):
        """Calculates the learning rate.

        Args:
            runner (object): The passed runner.
            base_lr (float): Base learning rate.

        Returns:
            float: Current learning rate.
        """
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        assert max_progress >= self.start

        if max_progress == self.start:
            return base_lr

        # Before 'start', fix lr; After 'start', linearly update lr.
        factor = (max(0, progress - self.start) // self.interval) / (
            (max_progress - self.start) // self.interval)
        return base_lr + (self.target_lr - base_lr) * factor


@HOOKS.register_module()
class ReduceLrUpdaterHook(LrUpdaterHook):
    """ReduceLROnPlateau Scheduler.

    Reduce learning rate when a metric has stopped improving. This scheduler
    reads a metrics quantity and if no improvement is seen for a 'patience'
    number of epochs, the learning rate is reduced.

    Args:
        val_metric (str, optional): Metrics to be evaluated. If val_metric is
            None, the metrics will be loss value. Default: None.
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
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        epoch_base_valid (None | Bool): Whether use epoch base valid.
            If `None`, follow `by_epoch` (inherited from `LrUpdaterHook`).
            Default: None.
    """

    def __init__(self,
                 val_metric: str = None,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: float = 0.,
                 eps: float = 1e-8,
                 verbose: bool = False,
                 epoch_base_valid=None,
                 **kwargs):

        self.val_metric = val_metric

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
        self.verbose = verbose
        self.last_epoch = 0
        self._init_is_better(self.mode)
        self._reset()

        super().__init__(**kwargs)
        if epoch_base_valid is None:
            self.epoch_base_valid = self.by_epoch
        else:
            self.epoch_base_valid = epoch_base_valid

    def get_lr(self, regular_lr, optimizer_name):
        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            if regular_lr - regular_lr * self.factor > self.eps:
                new_lr = max(regular_lr * self.factor, self.min_lr)
                if self.verbose:
                    print(f'Reducing learning rate of {optimizer_name} from '
                          f'{regular_lr:.4e} to {new_lr:.4e}.')
            else:
                new_lr = regular_lr
            return new_lr
        else:
            return regular_lr

    def get_regular_lr(self, runner):
        if not self.regular_lr:
            self.regular_lr = self.base_lr
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(_regular_lr, k)
                    for _regular_lr in self.regular_lr[k]
                ]
                lr_groups.update({k: _lr_group})
        else:
            lr_groups = [
                self.get_lr(_regular_lr, 'generator')
                for _regular_lr in self.regular_lr
            ]
        self.regular_lr = lr_groups
        return lr_groups

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

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        cur_epoch = runner.epoch
        if self.warmup is not None and self.warmup_by_epoch:
            if cur_epoch <= self.warmup_epochs:
                return
        # If val_metric is None, we check training loss to reduce learning
        # rate.
        if self.val_metric is None:
            current = runner.outputs['log_vars']['loss']
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0
            print(f'train_epoch--current {current:.6f} best {self.best:.6f}, '
                  f'num_bad_epochs {self.num_bad_epochs}, '
                  f'cooldown {self.in_cooldown} {self.cooldown_counter}')

    def after_train_iter(self, runner):
        if self.by_epoch:
            return
        cur_iter = runner.iter
        if self.warmup_epochs is not None and cur_iter <= self.warmup_iters:
            return
        # If val_metric is None, we check training loss to reduce learning
        # rate.
        if self.val_metric is None:
            current = runner.outputs['log_vars']['loss']
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0
            print(f'train_iter--current {current:.6f} best {self.best:.6f}, '
                  f'num_bad_epochs {self.num_bad_epochs}, '
                  f'cooldown {self.in_cooldown} {self.cooldown_counter}')

    def after_val_epoch(self, runner):
        if not self.by_epoch and not self.epoch_base_valid:
            return
        cur_epoch = runner.epoch
        if self.warmup is not None and self.warmup_by_epoch:
            if cur_epoch <= self.warmup_epochs:
                return
        # If val_metric is not None, we check its value to reduce learning
        # rate.
        if self.val_metric is not None:
            current = runner.log_buffer.output[self.val_metric]
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0
            print(f'val_epoch--current {current:.6f} best {self.best:.6f}, '
                  f'num_bad_epochs {self.num_bad_epochs}, '
                  f'cooldown {self.in_cooldown} {self.cooldown_counter}')

    def after_val_iter(self, runner):
        if self.by_epoch or self.epoch_base_valid:
            return
        cur_iter = runner.iter
        if self.warmup_epochs is not None and cur_iter <= self.warmup_iters:
            return
        # If val_metric is not None, we check its value to reduce learning
        # rate.
        if self.val_metric is not None:
            current = runner.eval_result[self.val_metric]
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0
            print(f'val_iter--current {current:.6f} best {self.best:.6f}, '
                  f'num_bad_epochs {self.num_bad_epochs}, '
                  f'cooldown {self.in_cooldown} {self.cooldown_counter}')
