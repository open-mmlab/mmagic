# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

from mmedit.utils import try_import
from .ddim_scheduler import EditDDIMScheduler
from .ddpm_scheduler import EditDDPMScheduler


def register_diffusers_schedulers() -> List[str]:
    """Register schedulers in ``diffusers.schedulers`` to the
    ``DIFFUSION_SCHEDULERS`` registry.

    Returns:
        List[str]: A list of registered DIFFUSION_SCHEDULERS' name.
    """

    import inspect

    from mmedit.registry import DIFFUSION_SCHEDULERS

    diffusers = try_import('diffusers')
    if diffusers is None:
        warnings.warn('Diffusion Schedulers are not registered as expect. '
                      'If you want to use diffusion models, '
                      'please install diffusers>=0.12.0.')
        return None

    DIFFUSERS_SCHEDULERS = []
    for module_name in dir(diffusers.schedulers):
        if module_name.startswith('Flax'):
            continue
        elif module_name.endswith('Scheduler'):
            _scheduler = getattr(diffusers.schedulers, module_name)
            if inspect.isclass(_scheduler):
                DIFFUSION_SCHEDULERS.register_module(
                    name=module_name, module=_scheduler)
                DIFFUSERS_SCHEDULERS.append(module_name)
    return DIFFUSERS_SCHEDULERS


REGISTERED_DIFFUSERS_SCHEDULERS = register_diffusers_schedulers()

__all__ = ['EditDDIMScheduler', 'EditDDPMScheduler']
