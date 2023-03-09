# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from .ddim_scheduler import DDIMScheduler
from .ddpm_scheduler import DDPMScheduler


def register_diffusers_schedulers() -> List[str]:
    """Register schedulers in ``diffusers.schedulers`` to the
    ``DIFFUSION_SCHEDULERS`` registry.

    Returns:
        List[str]: A list of registered DIFFUSION_SCHEDULERS' name.
    """

    import inspect

    import diffusers

    from mmedit.registry import DIFFUSION_SCHEDULERS

    DIFFUSERS_SCHEDULERS = []
    for module_name in dir(diffusers.schedulers):
        if module_name.startswith('Flax'):
            continue
        elif module_name.endswith('Scheduler'):
            _scheduler = getattr(diffusers.schedulers, module_name)
            if inspect.isclass(_scheduler):
                DIFFUSION_SCHEDULERS.register_module(
                    name='Diffusers' + module_name, module=_scheduler)
                DIFFUSERS_SCHEDULERS.append('Diffusers' + module_name)
    return DIFFUSERS_SCHEDULERS


REGISTERED_DIFFUSERS_SCHEDULERS = register_diffusers_schedulers()

__all__ = ['DDIMScheduler', 'DDPMScheduler']
