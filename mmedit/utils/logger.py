# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger
from mmengine.logging import print_log
from termcolor import colored


def print_colored_log(msg, level=logging.INFO, color='magenta'):
    """Print colored log with default logger.

    Args:
        msg (str): Message to log.
        level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.Log level,
            default to 'info'.
        color (str, optional): Color 'magenta'.
    """
    print_log(colored(msg, color), 'current', level)


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmedit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    # root logger name: mmedit
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger
