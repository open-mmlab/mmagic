# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.diffusion_schedulers.ddim_scheduler import EditDDIMScheduler


def test_ddim():
    modelout = torch.rand((1, 8, 32, 32))
    sample = torch.rand((1, 4, 32, 32))
    ddim = EditDDIMScheduler(
        num_train_timesteps=1000, variance_type='learned_range')
    ddim.set_timesteps(10)
    result = ddim.step(modelout, 980, sample)
    assert result['prev_sample'].shape == (1, 4, 32, 32)

    noise = torch.rand((1, 4, 32, 32))
    result = ddim.add_noise(sample, noise, 10)
    assert result.shape == (1, 4, 32, 32)

    assert len(ddim) == 1000


def test_ddim_init():
    ddim = EditDDIMScheduler(
        num_train_timesteps=1000, beta_schedule='scaled_linear')

    ddim = EditDDIMScheduler(
        num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    assert isinstance(ddim, EditDDIMScheduler)

    with pytest.raises(Exception):
        EditDDIMScheduler(num_train_timesteps=1000, beta_schedule='fake')


def test_ddim_step():
    modelout = torch.rand((1, 8, 32, 32))
    sample = torch.rand((1, 4, 32, 32))
    ddim = EditDDIMScheduler(
        num_train_timesteps=1000, variance_type='learned_range')
    with pytest.raises(Exception):
        ddim.step(modelout, 980, sample)

    ddim.set_timesteps(10)
    result = ddim.step(
        modelout, 980, sample, eta=1, use_clipped_model_output=True)
    assert result['prev_sample'].shape == (1, 4, 32, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
