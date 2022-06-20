# Copyright (c) OpenMMLab. All rights reserved.
from .cli import modify_args
from .logger import get_root_logger, print_colored_log
from .setup_env import setup_multi_processes

__all__ = [
    'modify_args',
    'print_colored_log',
    'get_root_logger',
    'setup_multi_processes',
]
