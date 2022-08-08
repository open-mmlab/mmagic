# Copyright (c) OpenMMLab. All rights reserved.
import logging

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
