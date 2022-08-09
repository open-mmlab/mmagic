# Copyright (c) OpenMMLab. All rights reserved.
from .cli import modify_args
# TODO replace with engine's API
from .logger import print_colored_log
from .setup_env import register_all_modules

__all__ = ['modify_args', 'print_colored_log', 'register_all_modules']
