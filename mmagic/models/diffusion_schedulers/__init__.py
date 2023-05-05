# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Any, List

from mmagic.utils import try_import
from .ddim_scheduler import EditDDIMScheduler
from .ddpm_scheduler import EditDDPMScheduler


class SchedulerWrapper:
    """Wrapper for schedulers from HuggingFace Diffusers. This wrapper will be
    set a attribute called `_scheduler_cls` by wrapping function and will be
    used to initialize the model structure.

    Example:
    >>> 1. Load pretrained model from HuggingFace Space.
    >>> config = dict(
    >>>     type='DDPMScheduler',
    >>>     from_pretrained='lllyasviel/sd-controlnet-canny',
    >>>     subfolder='scheduler')
    >>> ddpm_scheduler = DIFFUSION_SCHEDULERS.build(config)

    >>> 2. Initialize model with own defined arguments
    >>> config = dict(
    >>>     type='EulerDiscreteScheduler',
    >>>     num_train_timesteps=2000,
    >>>     beta_schedule='scaled_linear')
    >>> euler_scheduler = DIFFUSION_SCHEDULERS.build(config)

    Args:
        from_pretrained (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Please refers to
            `diffusers.model.modeling_utils.ModelMixin.from_pretrained`
            for more detail. Defaults to None.

        *args, **kwargs: If `from_pretrained` is passed, *args and **kwargs
            will be passed to `from_pretrained` function. Otherwise, *args
            and **kwargs will be used to initialize the model by
            `self._module_cls(*args, **kwargs)`.
    """

    def __init__(self,
                 from_pretrained=None,
                 from_config=None,
                 *args,
                 **kwargs):

        scheduler_cls = self._scheduler_cls

        self._from_pretrained = from_pretrained
        self._from_config = from_config
        if self._from_pretrained:
            self.scheduler = scheduler_cls.from_pretrained(
                from_pretrained, *args, **kwargs)
        elif self._from_config:
            self.scheduler = scheduler_cls.from_config(from_config, *args,
                                                       **kwargs)
        else:
            self.scheduler = scheduler_cls(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """This function provide a way to access the attributes of the wrapped
        scheduler.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The got attribute.
        """

        try:
            return getattr(self.scheduler, name)
        except AttributeError:
            raise AttributeError(f'{name} cannot be found in both '
                                 f'\'{self.__class__.__name__}\' and '
                                 f'\'{self.__class__.__name__}.scheduler\'.')

    def __repr__(self):
        """The representation of the wrapper."""
        s = super().__repr__()
        prefix = f'Wrapped Scheduler Class: {self._scheduler_cls}\n'
        prefix += f'Wrapped Scheduler Name: {self._scheduler_name}\n'
        if self._from_pretrained:
            prefix += f'From Pretrained: {self._from_pretrained}\n'
        if self._from_config:
            prefix += f'From Config: {self._from_config}\n'
        s = prefix + s
        return s


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

    from mmagic.registry import DIFFUSION_SCHEDULERS

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
