# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """Create a beta schedule that discretized the given alpha_t_bar
    function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py#L49 # noqa
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2)**2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)
