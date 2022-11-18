# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import MessageHub
from mmengine.optim import LinearLR

from mmedit.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class LinearLrInterval(LinearLR):
    """Linear learning rate scheduler for image generation.

    In the beginning, the learning rate is 'start_factor' defined in mmengine.
    We give a target learning rate 'end_factor' and a start point 'begin'.
    If :attr:self.by_epoch is True, 'begin' is calculated by epoch, otherwise,
    calculated by iteration." Before 'begin', we fix learning rate as
    'start_factor'; After 'begin', we linearly update learning rate to
    'end_factor'.

    Args:
        interval (int): The interval to update the learning rate. Default: 1.
    """

    def __init__(self, *args, interval=1, **kwargs):
        self.interval = interval
        super().__init__(*args, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                group[self.param_name] * self.start_factor
                for group in self.optimizer.param_groups
            ]

        message_hub = MessageHub.get_current_instance()
        if self.by_epoch:
            progress = message_hub.get_info('epoch')
        else:
            progress = message_hub.get_info('iter')

        max_progress = self.end

        factor = (max(0, progress - self.begin) // self.interval) / (
            (max_progress - self.begin) // self.interval)

        return [
            self.start_factor + (self.end_factor - self.start_factor) * factor
            for group in self.optimizer.param_groups
        ]
