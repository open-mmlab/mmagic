# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Optional

import torch.nn as nn
from mmengine.model import is_model_wrapper
from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapperDict

from mmedit.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class PGGANOptimWrapperConstructor:
    """OptimizerConstructor for PGGAN models. Set optimizers for each
    stage of PGGAN. All submodule must be contained in a
    :class:~`torch.nn.ModuleList` named 'blocks'. And we access each submodule
    by `MODEL.blocks[SCALE]`, where `MODLE` is generator or discriminator, and
    the scale is the index of the resolution scale.

    More detail about the resolution scale and naming rule please refers to
    :class:~`mmgen.models.PGGANGenerator` and
    :class:~`mmgen.models.PGGANDiscriminator`.

    Example:
        >>> # build PGGAN model
        >>> model = dict(
        >>>     type='ProgressiveGrowingGAN',
        >>>     data_preprocessor=dict(type='GANDataPreprocessor'),
        >>>     noise_size=512,
        >>>     generator=dict(type='PGGANGenerator', out_scale=1024,
        >>>                    noise_size=512),
        >>>     discriminator=dict(type='PGGANDiscriminator', in_scale=1024),
        >>>     nkimgs_per_scale={
        >>>         '4': 600,
        >>>         '8': 1200,
        >>>         '16': 1200,
        >>>         '32': 1200,
        >>>         '64': 1200,
        >>>         '128': 1200,
        >>>         '256': 1200,
        >>>         '512': 1200,
        >>>         '1024': 12000,
        >>>     },
        >>>     transition_kimgs=600,
        >>>     ema_config=dict(interval=1))
        >>> pggan = MODELS.build(model)
        >>> # build constructor
        >>> optim_wrapper = dict(
        >>>     generator=dict(optimizer=dict(type='Adam', lr=0.001,
        >>>                                   betas=(0., 0.99))),
        >>>     discriminator=dict(
        >>>         optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
        >>>     lr_schedule=dict(
        >>>         generator={
        >>>             '128': 0.0015,
        >>>             '256': 0.002,
        >>>             '512': 0.003,
        >>>             '1024': 0.003
        >>>         },
        >>>         discriminator={
        >>>             '128': 0.0015,
        >>>             '256': 0.002,
        >>>             '512': 0.003,
        >>>             '1024': 0.003
        >>>         }))
        >>> optim_wrapper_dict_builder = PGGANOptimWrapperConstructor(
        >>>     optim_wrapper)
        >>> # build optim wrapper dict
        >>> optim_wrapper_dict = optim_wrapper_dict_builder(pggan)

    Args:
        optim_wrapper_cfg (dict): Config of the optimizer wrapper.
        paramwise_cfg (Optional[dict]): Parameter-wise options.
    """

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optim_wrapper_cfg)}')
        assert paramwise_cfg is None, (
            'parawise_cfg should be set in each optimizer separately')
        self.optim_cfg = deepcopy(optim_wrapper_cfg)

        self.reset_optim = self.optim_cfg.pop('reset_optim_for_new_scale',
                                              True)
        print(self.reset_optim)
        self.lr_schedule = self.optim_cfg.pop('lr_schedule', dict())
        self.constructors = {}

        for key, cfg in self.optim_cfg.items():
            cfg_ = cfg.copy()
            paramwise_cfg_ = cfg_.pop('paramwise_cfg', None)
            self.constructors[key] = DefaultOptimWrapperConstructor(
                cfg_, paramwise_cfg_)

    def __call__(self, module: nn.Module) -> OptimWrapperDict:
        """Build optimizer and return a optimizerwrapperdict."""
        optimizers = {}
        if is_model_wrapper(module):
            module = module.module

        # module.scales: [int, int]
        scales = [s[0] for s in module.scales]

        for key, base_cfg in self.optim_cfg.items():
            submodule = module._modules[key]

            cfg_ = base_cfg.copy()
            base_lr = cfg_['optimizer']['lr']
            paramwise_cfg_ = base_cfg.pop('paramwise_cfg', None)

            default_constructor = self.constructors[key]
            default_optimizer = default_constructor(submodule)
            for idx, scale in enumerate(scales):
                if self.reset_optim:
                    scale_cfg = cfg_.copy()
                    scale_lr = self.lr_schedule[key].get(str(scale), base_lr)
                    scale_cfg['optimizer']['lr'] = scale_lr
                    constructor = DefaultOptimWrapperConstructor(
                        scale_cfg, paramwise_cfg_)
                    optimizers[f'{key}_{scale}'] = constructor(submodule)
                else:
                    optimizers[f'{key}_{scale}'] = default_optimizer

        optimizers = OptimWrapperDict(**optimizers)
        return optimizers
