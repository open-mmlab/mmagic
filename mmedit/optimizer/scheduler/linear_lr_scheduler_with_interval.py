# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim import LinearLR

from mmedit.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class LinearLRWithInterval(LinearLR):
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
        end (int): Step at which to stop updating the learning rate.
        interval (int): The interval to update the learning rate. Default: 1.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        start_factor (float): The number we multiply learning rate in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 1.
        end_factor (float): The number we multiply learning rate at the end
            of linear changing process. Defaults to 0.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer,
                 end,
                 interval=1,
                 begin=0,
                 start_factor=1.,
                 end_factor=0.,
                 **kwargs):

        self.interval = interval

        super().__init__(
            optimizer=optimizer,
            end=end,
            begin=begin,
            start_factor=start_factor,
            end_factor=end_factor,
            **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""

        if self.last_step == 0:
            return [
                group[self.param_name] * self.start_factor
                for group in self.optimizer.param_groups
            ]
        elif self.last_step % self.interval != 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        factor = (1. + (self.end_factor - self.start_factor) /
                  (self.total_iters / self.interval * self.start_factor +
                   (self.last_step / self.interval - 1) *
                   (self.end_factor - self.start_factor)))

        return [
            group[self.param_name] * factor
            for group in self.optimizer.param_groups
        ]
