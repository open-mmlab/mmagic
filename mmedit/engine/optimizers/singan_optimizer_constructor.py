# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapperDict

from mmedit.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class SinGANOptimWrapperConstructor:
    """OptimizerConstructor for SinGAN models. Set optimizers for each
    submodule of SinGAN. All submodule must be contained in a
    :class:~`torch.nn.ModuleList` named 'blocks'. And we access each submodule
    by `MODEL.blocks[SCALE]`, where `MODLE` is generator or discriminator, and
    the scale is the index of the resolution scale.

    More detail about the resolution scale and naming rule please refers to
    :class:~`mmgen.models.SinGANMultiScaleGenerator` and
    :class:~`mmgen.models.SinGANMultiScaleDiscriminator`.

    Example:
        >>> # build SinGAN model
        >>> model = dict(
        >>>     type='SinGAN',
        >>>     data_preprocessor=dict(
        >>>         type='GANDataPreprocessor',
        >>>         non_image_keys=['input_sample']),
        >>>     generator=dict(
        >>>         type='SinGANMultiScaleGenerator',
        >>>         in_channels=3,
        >>>         out_channels=3,
        >>>         num_scales=2),
        >>>     discriminator=dict(
        >>>         type='SinGANMultiScaleDiscriminator',
        >>>         in_channels=3,
        >>>         num_scales=3))
        >>> singan = MODELS.build(model)
        >>> # build constructor
        >>> optim_wrapper = dict(
        >>>     generator=dict(optimizer=dict(type='Adam', lr=0.0005,
        >>>                                   betas=(0.5, 0.999))),
        >>>     discriminator=dict(
        >>>         optimizer=dict(type='Adam', lr=0.0005,
        >>>                        betas=(0.5, 0.999))))
        >>> optim_wrapper_dict_builder = SinGANOptimWrapperConstructor(
        >>>     optim_wrapper)
        >>> # build optim wrapper dict
        >>> optim_wrapper_dict = optim_wrapper_dict_builder(singan)

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
        self.optim_cfg = optim_wrapper_cfg
        self.constructors = {}
        for key, cfg in self.optim_cfg.items():
            cfg_ = cfg.copy()
            paramwise_cfg_ = cfg_.pop('paramwise_cfg', None)
            self.constructors[key] = DefaultOptimWrapperConstructor(
                cfg_, paramwise_cfg_)

    def __call__(self, module: nn.Module) -> OptimWrapperDict:
        """Build optimizer and return a optimizerwrapperdict."""
        optimizers = {}
        if hasattr(module, 'module'):
            module = module.module
        num_scales = module.num_scales

        for key, constructor in self.constructors.items():
            for idx in range(num_scales + 1):
                submodule = module._modules[key]
                if hasattr(submodule, 'module'):
                    submodule = submodule.module
                optimizers[f'{key}_{idx}'] = constructor(submodule.blocks[idx])
        optimizers = OptimWrapperDict(**optimizers)
        return optimizers
