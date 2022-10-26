# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from torch import Tensor


def gather_log_vars(log_vars_list: List[Dict[str,
                                             Tensor]]) -> Dict[str, Tensor]:
    """Gather a list of log_vars.
    Args:
        log_vars_list: List[Dict[str, Tensor]]

    Returns:
        Dict[str, Tensor]
    """
    if len(log_vars_list) == 1:
        return log_vars_list[0]

    log_keys = log_vars_list[0].keys()

    log_vars = dict()
    for k in log_keys:
        assert all([k in log_vars for log_vars in log_vars_list
                    ]), (f'\'{k}\' not in some of the \'log_vars\'.')
        log_vars[k] = torch.mean(
            torch.stack([log_vars[k] for log_vars in log_vars_list], dim=0))

    return log_vars
