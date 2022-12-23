# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.editors.ddpm.ddpm_scheduler import DDPMScheduler


def test_ddpm():
    modelout = torch.rand((1, 8, 32, 32))
    sample = torch.rand((1, 4, 32, 32))
    ddpm = DDPMScheduler(
        num_train_timesteps=1000, variance_type='learned_range')
    result = ddpm.step(modelout, 980, sample)
    assert result['prev_sample'].shape == (1, 4, 32, 32)

    ddpm.set_timesteps(100)

    predicted_variance = torch.tensor(1)
    ddpm._get_variance(t=0, predicted_variance=predicted_variance)
    ddpm._get_variance(t=1, variance_type='fixed_large')
    ddpm._get_variance(t=1, variance_type='fixed_large_log')
    ddpm._get_variance(t=1, variance_type='learned')

    with pytest.raises(Exception):
        ddpm.training_loss(1, 2, 3)

    with pytest.raises(Exception):
        ddpm.sample_timestep()

    steps = len(ddpm)
    assert steps == 1000


if __name__ == '__main__':
    test_ddpm()
