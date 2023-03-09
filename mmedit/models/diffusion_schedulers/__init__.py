# Copyright (c) OpenMMLab. All rights reserved.
from .ddim_scheduler import EditDDIMScheduler
from .ddpm_scheduler import EditDDPMScheduler

__all__ = ['EditDDIMScheduler', 'EditDDPMScheduler']

import inspect
from typing import List

import diffusers

from mmedit.registry import DIFFUSION_SCHEDULERS


def register_diffusers_schedulers() -> List[str]:
    """Register schedulers in ``diffusers.schedulers`` to the
    ``DIFFUSION_SCHEDULERS`` registry.

    Returns:
        List[str]: A list of registered DIFFUSION_SCHEDULERS' name.
    """
    DIFFUSERS_SCHEDULERS = []
    for module_name in dir(diffusers.schedulers):
        if module_name.startswith('Flax'):
            continue
        elif module_name.endswith('Scheduler'):
            _scheduler = getattr(diffusers.schedulers, module_name)
            if inspect.isclass(_scheduler):
                DIFFUSION_SCHEDULERS.register_module(name='Diffusers'+module_name, module=_scheduler)
                DIFFUSERS_SCHEDULERS.append('Diffusers' + module_name)
    return DIFFUSERS_SCHEDULERS


DIFFUSERS_SCHEDULERS = register_diffusers_schedulers()
