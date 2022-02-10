# Copyright (c) OpenMMLab. All rights reserved.
from .logger import get_root_logger
from .setup_env import setup_multi_processes
from .testing import dict_to_cuda

__all__ = ['get_root_logger', 'dict_to_cuda', 'setup_multi_processes']
