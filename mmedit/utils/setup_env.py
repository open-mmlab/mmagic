# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


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
    import mmedit.visualizer  # noqa: F401,F403

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
