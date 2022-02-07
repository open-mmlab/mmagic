# Copyright (c) OpenMMLab. All rights reserved.
from .logger import get_root_logger
from .setup_env import setup_multi_processes

__all__ = ['get_root_logger', setup_multi_processes]
