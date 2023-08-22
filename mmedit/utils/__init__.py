# Copyright (c) OpenMMLab. All rights reserved.
from .cli import modify_args
from .logger import get_root_logger
from .misc import deprecated_function
from .setup_env import setup_multi_processes

__all__ = [
    'get_root_logger', 'setup_multi_processes', 'modify_args',
    'deprecated_function'
]
