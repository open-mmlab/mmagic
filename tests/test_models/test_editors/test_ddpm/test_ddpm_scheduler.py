# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.ddpm.ddpm_scheduler import DDPMScheduler


def test_ddpm():
    modelout = torch.rand((1, 8, 32, 32))
    sample = torch.rand((1, 4, 32, 32))
    ddpm = DDPMScheduler(
        num_train_timesteps=1000, variance_type='learned_range')
    result = ddpm.step(modelout, 980, sample)
    assert result['prev_sample'].shape == (1, 4, 32, 32)


if __name__ == '__main__':
    test_ddpm()
