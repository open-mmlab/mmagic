# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Tuple, Union

import torch.nn as nn
from mmengine import print_log
from mmengine.optim import (DefaultOptimWrapperConstructor, OptimWrapper,
                            OptimWrapperDict)

from mmagic.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                             OPTIMIZERS)


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MultiOptimWrapperConstructor:
    """OptimizerConstructor for GAN models. This class construct optimizer for
    the submodules of the model separately, and return a
    :class:`mmengine.optim.OptimWrapperDict` or
    :class:`mmengine.optim.OptimWrapper`.

    Example 1: Build multi optimizers (e.g., GANs):
        >>> # build GAN model
        >>> model = dict(
        >>>     type='GANModel',
        >>>     num_classes=10,
        >>>     generator=dict(type='Generator'),
        >>>     discriminator=dict(type='Discriminator'))
        >>> gan_model = MODELS.build(model)
        >>> # build constructor
        >>> optim_wrapper = dict(
        >>>     generator=dict(
        >>>         type='OptimWrapper',
        >>>         accumulative_counts=1,
        >>>         optimizer=dict(type='Adam', lr=0.0002,
        >>>                        betas=(0.5, 0.999))),
        >>>     discriminator=dict(
        >>>         type='OptimWrapper',
        >>>         accumulative_counts=1,
        >>>         optimizer=dict(type='Adam', lr=0.0002,
        >>>                            betas=(0.5, 0.999))))
        >>> optim_dict_builder = MultiOptimWrapperConstructor(optim_wrapper)
        >>> # build optim wrapper dict
        >>> optim_wrapper_dict = optim_dict_builder(gan_model)

    Example 2: Build multi optimizers for specific submodules:
        >>> # build model
        >>> class GAN(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.generator = nn.Conv2d(3, 3, 1)
        >>>         self.discriminator = nn.Conv2d(3, 3, 1)
        >>> class TextEncoder(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.embedding = nn.Embedding(100, 100)
        >>> class ToyModel(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.m1 = GAN()
        >>>         self.m2 = nn.Conv2d(3, 3, 1)
        >>>         self.m3 = nn.Linear(2, 2)
        >>>         self.text_encoder = TextEncoder()
        >>> model = ToyModel()
        >>> # build constructor
        >>> optim_wrapper = {
        >>>     '.*embedding': {
        >>>         'type': 'OptimWrapper',
        >>>         'optimizer': {
        >>>             'type': 'Adam',
        >>>             'lr': 1e-4,
        >>>             'betas': (0.9, 0.99)
        >>>         }
        >>>     },
        >>>     'm1.generator': {
        >>>         'type': 'OptimWrapper',
        >>>         'optimizer': {
        >>>             'type': 'Adam',
        >>>             'lr': 1e-5,
        >>>             'betas': (0.9, 0.99)
        >>>         }
        >>>     },
        >>>     'm2': {
        >>>         'type': 'OptimWrapper',
        >>>         'optimizer': {
        >>>             'type': 'Adam',
        >>>             'lr': 1e-5,
        >>>         }
        >>>     }
        >>> }
        >>> optim_dict_builder = MultiOptimWrapperConstructor(optim_wrapper)
        >>> # build optim wrapper dict
        >>> optim_wrapper_dict = optim_dict_builder(model)

    Example 3: Build a single optimizer for multi modules (e.g., DreamBooth):
        >>> # build StableDiffusion model
        >>> model = dict(
        >>>     type='StableDiffusion',
        >>>     unet=dict(type='unet'),
        >>>     vae=dict(type='vae'),
                text_encoder=dict(type='text_encoder'))
        >>> diffusion_model = MODELS.build(model)
        >>> # build constructor
        >>> optim_wrapper = dict(
        >>>     modules=['unet', 'text_encoder']
        >>>     optimizer=dict(type='Adam', lr=0.0002),
        >>>     accumulative_counts=1)
        >>> optim_dict_builder = MultiOptimWrapperConstructor(optim_wrapper)
        >>> # build optim wrapper dict
        >>> optim_wrapper_dict = optim_dict_builder(diffusion_model)

    Args:
        optim_wrapper_cfg_dict (dict): Config of the optimizer wrapper.
        paramwise_cfg (dict): Config of parameter-wise settings. Default: None.
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg=None):

        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optim_wrapper_cfg)}')
        assert paramwise_cfg is None, (
            'paramwise_cfg should be set in each optimizer separately')
        self.optim_cfg = optim_wrapper_cfg

        if 'modules' in optim_wrapper_cfg:
            # single optimizer with multi param groups
            cfg_ = optim_wrapper_cfg.copy()
            self.modules = cfg_.pop('modules')
            paramwise_cfg_ = cfg_.pop('paramwise_cfg', None)
            self.constructors = DefaultOptimWrapperConstructor(
                cfg_, paramwise_cfg_)
        else:
            self.constructors = {}
            self.modules = {}
            for key, cfg in self.optim_cfg.items():
                cfg_ = cfg.copy()
                if 'modules' in cfg_:
                    self.modules[key] = cfg_.pop('modules')
                paramwise_cfg_ = cfg_.pop('paramwise_cfg', None)
                self.constructors[key] = DefaultOptimWrapperConstructor(
                    cfg_, paramwise_cfg_)

    def __call__(self,
                 module: nn.Module) -> Union[OptimWrapperDict, OptimWrapper]:
        """Build optimizer and return a optimizer_wrapper_dict."""

        optimizers = {}
        if hasattr(module, 'module'):
            module = module.module
        if isinstance(self.constructors, dict):
            for key, constructor in self.constructors.items():
                module_names = self.modules[key] if self.modules else key
                if (isinstance(module_names, str)
                        and module_names in module._modules):
                    optimizers[key] = constructor(
                        module._modules[module_names])
                    optim_wrapper_cfg = constructor.optimizer_cfg
                    print_log(
                        f'Add to optimizer \'{key}\' '
                        f'({optim_wrapper_cfg}): \'{key}\'.', 'current')
                else:

                    assert not constructor.paramwise_cfg, (
                        'Do not support paramwise_cfg for multi module '
                        'optimizer.')

                    params, found_names = get_params_by_names(
                        module, module_names)
                    # build optimizer
                    optimizer_cfg = constructor.optimizer_cfg.copy()
                    optimizer_cfg['params'] = params
                    optimizer = OPTIMIZERS.build(optimizer_cfg)

                    # build optimizer wrapper
                    optim_wrapper_cfg = constructor.optim_wrapper_cfg.copy()
                    optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
                    optim_wrapper = OPTIM_WRAPPERS.build(
                        optim_wrapper_cfg,
                        default_args=dict(optimizer=optimizer))

                    for name in found_names:
                        print_log(
                            f'Add to optimizer \'{key}\' '
                            f'({constructor.optimizer_cfg}): \'{name}\'.',
                            'current')

                    optimizers[key] = optim_wrapper

            return OptimWrapperDict(**optimizers)

        else:
            params, found_names = get_params_by_names(module, self.modules)

            constructor = self.constructors
            assert not constructor.paramwise_cfg, (
                'Do not support paramwise_cfg for multi parameters')

            optimizer_cfg = constructor.optimizer_cfg.copy()
            optimizer_cfg['params'] = params
            optimizer = OPTIMIZERS.build(optimizer_cfg)
            for name in found_names:
                print_log(
                    f'Add to optimizer ({constructor.optimizer_cfg}): '
                    f'\'{name}\'.', 'current')

            # build optimizer wrapper
            optim_wrapper_cfg = constructor.optim_wrapper_cfg.copy()
            optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
            optim_wrapper = OPTIM_WRAPPERS.build(
                optim_wrapper_cfg, default_args=dict(optimizer=optimizer))

            return optim_wrapper


def get_params_by_names(module: nn.Module,
                        names: Union[str, list]) -> Tuple[list, list]:
    """Support two kinds of name matching:
        1. matching name from **first-level** submodule.
        2. matching name by `re.fullmatch`.

    Args:
        module (nn.Module): The module to get parameters.
        names (Union[str, list]): The name or a list of names of the
            submodule parameters.

    Returns:
        Tuple[list]: A list of parameters and corresponding name for logging.
    """

    if not isinstance(names, list):
        names = [names]

    params = []
    found_names = []
    for name in names:
        if name in module._modules:
            params.extend(module._modules[name].parameters())
            found_names.append(name)
        else:
            for n, m in module.named_modules():
                if re.fullmatch(name, n):
                    params.extend(m.parameters())
                    found_names.append(n)
    return params, found_names
