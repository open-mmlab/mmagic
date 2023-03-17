# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

from mmedit.utils import try_import
from .ddim_scheduler import EditDDIMScheduler
from .ddpm_scheduler import EditDDPMScheduler


class SchedulerWrapper:

    def __init__(self, from_pretrained=None, *args, **kwargs):

        scheduler_cls = self._scheduler_cls

        self._from_pretrained = from_pretrained
        if self._from_pretrained:
            self.scheduler = scheduler_cls.from_pretrained(
                from_pretrained, *args, **kwargs)
        else:
            self.scheduler = scheduler_cls(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return getattr(self.scheduler, name)
        except AttributeError:
            raise AttributeError('\'name\' cannot be found in both '
                                 f'\'{self.__class__.__name__}\' and '
                                 f'\'{self.__class__.__name__}.scheduler\'.')


def register_diffusers_schedulers() -> List[str]:
    """Register schedulers in ``diffusers.schedulers`` to the
    ``DIFFUSION_SCHEDULERS`` registry. Specifically, the registered schedulers
    from diffusers define the methodology for iteratively adding noise to an
    image or for updating a sample based on model outputs. See more details
    about schedulers in diffusers here:
    https://huggingface.co/docs/diffusers/api/schedulers/overview.

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

    def gen_wrapped_cls(scheduler, scheduler_name):
        return type(
            scheduler_name, (SchedulerWrapper, ),
            dict(
                _scheduler_cls=scheduler,
                _scheduler_name=scheduler_name,
                __module__=__name__))

    DIFFUSERS_SCHEDULERS = []
    for module_name in dir(diffusers.schedulers):
        if module_name.startswith('Flax'):
            continue
        elif module_name.endswith('Scheduler'):
            _scheduler = getattr(diffusers.schedulers, module_name)
            if inspect.isclass(_scheduler):
                wrapped_scheduler = gen_wrapped_cls(_scheduler, module_name)
                DIFFUSION_SCHEDULERS.register_module(
                    name=module_name, module=wrapped_scheduler)
                DIFFUSERS_SCHEDULERS.append(module_name)
    return DIFFUSERS_SCHEDULERS


REGISTERED_DIFFUSERS_SCHEDULERS = register_diffusers_schedulers()

__all__ = ['EditDDIMScheduler', 'EditDDPMScheduler']
