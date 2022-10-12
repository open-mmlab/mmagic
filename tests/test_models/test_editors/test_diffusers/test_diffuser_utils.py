# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.editors.diffusers import betas_for_alpha_bar

def test_betas_for_alpha_bar(self):
    num_diffusion_timesteps = 1000
    betas = betas_for_alpha_bar(num_diffusion_timesteps)
    assert betas.shape == (1000)