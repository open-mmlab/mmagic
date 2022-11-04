# Copyright (c) OpenMMLab. All rights reserved.
# In my mind, the learning rate scheduler in AirNet can be considered as
# a gamma-changed MultiStepLR
# Write it here
# Then, prepare datasets for original airnet
# Then, or move airnet there
# from mmengine import MessageHub
from mmengine.optim import MultiStepLR

from mmedit.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class MultiMultiStepLR(MultiStepLR):
    """Decays the specified learning rate in each parameter group by gamma once
    the number of epoch reaches one of the milestones.

    Notice that if the gamma is a list. The learning rate will be decayed
    by the corresponding gamma for each milestone.
    Thus the gamma length should be equal to the length of the milestones.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """

    def __init__(self, optimizer, gamma, *args, **kwargs):
        super().__init__(optimizer, *args, **kwargs)
