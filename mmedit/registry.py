# Copyright (c) OpenMMLab. All rights reserved.
"""Registry Module.

MMEditing provides 17 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine import registry
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

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=registry.MODELS)
BACKBONES = MODELS
COMPONENTS = MODELS
LOSSES = MODELS
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper', parent=registry.MODEL_WRAPPERS)
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer', parent=registry.WEIGHT_INITIALIZERS)

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer', parent=registry.OPTIMIZERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor', parent=registry.OPTIM_WRAPPER_CONSTRUCTORS)
# mangage all kinds of parameter schedulers like `MultiStepLR`
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
