# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models import DDPMDiffuser

def test_ddpm_diffuser(self):
    ddpm_diffuser = DDPMDiffuser(variance_type='learned_range', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    # variance type
    ddpm_diffuser = DDPMDiffuser(variance_type='learned', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    ddpm_diffuser = DDPMDiffuser(variance_type='fixed_small', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    ddpm_diffuser = DDPMDiffuser(variance_type='fixed_small_log', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    ddpm_diffuser = DDPMDiffuser(variance_type='fixed_large', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    ddpm_diffuser = DDPMDiffuser(variance_type='fixed_large_log', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    # beta schedule
    ddpm_diffuser = DDPMDiffuser(variance_type='learned_range', beta_schedule='scaled_linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    ddpm_diffuser = DDPMDiffuser(variance_type='learned_range', beta_schedule='squaredcos_cap_v2')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddpm_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)