# Copyright (c) Open-MMLab.
from typing import Any, Dict

import torch


def dict_to_cuda(cpu_dict_data: Dict[Any, Any]) -> Dict[Any, Any]:
    """Move all tensor item in a dict to cuda, for testing in GPU.

    Args:
        cpu_dict_data (dict): Target data.

    Returns:
        dict: A dict whose tensor items are in cuda.
    """

    gpu_dict_data = dict()
    for key, value in cpu_dict_data.items():
        if isinstance(value, torch.Tensor):
            gpu_dict_data[key] = value.cuda()
        else:
            gpu_dict_data[key] = value

    return gpu_dict_data
