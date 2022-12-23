# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.editors.ddim.ddim_scheduler import DDIMScheduler


def test_ddim():
    modelout = torch.rand((1, 8, 32, 32))
    sample = torch.rand((1, 4, 32, 32))
    ddim = DDIMScheduler(
        num_train_timesteps=1000, variance_type='learned_range')
    ddim.set_timesteps(10)
    result = ddim.step(modelout, 980, sample)
    assert result['prev_sample'].shape == (1, 4, 32, 32)

    noise = torch.rand((1, 4, 32, 32))
    result = ddim.add_noise(sample, noise, 10)
    assert result.shape == (1, 4, 32, 32)


def test_ddim_init():
    ddim = DDIMScheduler(
        num_train_timesteps=1000, beta_schedule='scaled_linear')

    ddim = DDIMScheduler(
        num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    assert isinstance(ddim, DDIMScheduler)

    with pytest.raises(Exception):
        DDIMScheduler(num_train_timesteps=1000, beta_schedule='fake')


if __name__ == '__main__':
    test_ddim()
