# Copyright (c) OpenMMLab. All rights reserved.
"""Registries and utilities in MMEditing.

MMEditing provides 17 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

import datetime
import warnings

from mmengine import DefaultScope, registry
from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=registry.RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=registry.RUNNER_CONSTRUCTORS)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=registry.LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook', parent=registry.HOOKS)

# manage data-related modules
DATASETS = Registry('dataset', parent=registry.DATASETS)
DATA_SAMPLERS = Registry('data sampler', parent=registry.DATA_SAMPLERS)
TRANSFORMS = Registry('transform', parent=registry.TRANSFORMS)

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=registry.MODELS)
BACKBONES = COMPONENTS = LOSSES = MODELS
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper', parent=registry.MODEL_WRAPPERS)
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer', parent=registry.WEIGHT_INITIALIZERS)

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer', parent=registry.OPTIMIZERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=registry.OPTIM_WRAPPER_CONSTRUCTORS)
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', parent=registry.PARAM_SCHEDULERS)
# manage all kinds of metrics
METRICS = Registry('metric', parent=registry.METRICS)

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util', parent=registry.TASK_UTILS)

# manage visualizer
VISUALIZERS = Registry('visualizer', parent=registry.VISUALIZERS)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=registry.VISBACKENDS)

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor', parent=registry.LOG_PROCESSORS)

# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optim_wrapper', parent=registry.OPTIM_WRAPPERS)


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmedit into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmedit default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmedit`, and all registries will build modules from
            mmedit's registry node.
            To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmedit.datasets  # noqa: F401,F403
    import mmedit.hooks  # noqa: F401,F403
    import mmedit.metrics  # noqa: F401,F403
    import mmedit.models  # noqa: F401,F403
    import mmedit.optimizer  # noqa: F401,F403
    import mmedit.transforms  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
            or not DefaultScope.check_instance_created('mmedit')
        if never_created:
            DefaultScope.get_instance('mmedit', scope_name='mmedit')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmedit':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmedit", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmedit". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmedit-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmedit')
