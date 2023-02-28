# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import importlib
import warnings
from types import ModuleType
from typing import Optional

from mmengine import Config, DefaultScope
from mmengine.config import ConfigDict
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.runner import set_random_seed as set_random_seed_engine

from mmedit.registry import MODELS


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
    import mmedit.engine  # noqa: F401,F403
    import mmedit.evaluation  # noqa: F401,F403
    import mmedit.models  # noqa: F401,F403
    import mmedit.visualization  # noqa: F401,F403

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


def try_import(name: str) -> Optional[ModuleType]:
    """Try to import a module.

    Args:
        name (str): Specifies what module to import in absolute or relative
            terms (e.g. either pkg.mod or ..mod).
    Returns:
        ModuleType or None: If importing successfully, returns the imported
        module, otherwise returns None.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.test_cfg.metrics = None
    delete_cfg(config.model, 'init_cfg')

    init_default_scope(config.get('default_scope', 'mmedit'))

    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


def set_random_seed(seed, deterministic=False, use_rank_shift=True):
    """Set random seed.

    In this function, we just modify the default behavior of the similar
    function defined in MMCV.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: True.
    """
    set_random_seed_engine(seed, deterministic, use_rank_shift)


def delete_cfg(cfg, key='init_cfg'):
    """Delete key from config object.

    Args:
        cfg (str or :obj:`mmengine.Config`): Config object.
        key (str): Which key to delete.
    """

    if key in cfg:
        cfg.pop(key)
    for _key in cfg.keys():
        if isinstance(cfg[_key], ConfigDict):
            delete_cfg(cfg[_key], key)
