# Copyright (c) OpenMMLab. All rights reserved.
from .cli import modify_args
# TODO replace with engine's API
from .logger import get_root_logger, print_colored_log

__all__ = [
    'modify_args',
    'print_colored_log',
    'get_root_logger',
]
