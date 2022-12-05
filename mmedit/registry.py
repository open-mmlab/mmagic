# Copyright (c) OpenMMLab. All rights reserved.
"""Registries and utilities in MMEditing.

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

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=registry.MODELS)
MODULES = BACKBONES = COMPONENTS = LOSSES = MODELS
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
# manage all kinds of evaluators
EVALUATORS = Registry('evaluator', parent=registry.EVALUATOR)

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

# manage diffusion_schedulers
DIFFUSION_SCHEDULERS = Registry('diffusion scheduler')
