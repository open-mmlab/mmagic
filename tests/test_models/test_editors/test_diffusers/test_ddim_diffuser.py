# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models import DDIMDiffuser

def test_ddim_diffuser(self):
    ddim_diffuser = DDIMDiffser(variance_type='learned_range', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddim_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    # variance type
    ddim_diffuser = DDIMDiffser(variance_type='learned', beta_schedule='linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddim_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    # beta schedule
    ddim_diffuser = DDIMDiffser(variance_type='learned_range', beta_schedule='scaled_linear')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddim_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)

    ddim_diffuser = DDIMDiffser(variance_type='learned_range', beta_schedule='squaredcos_cap_v2')
    sample = torch.randn((1, 3, 64, 64))
    model_output = torch.randn((1, 6, 64, 64))
    timestep = torch.randint(0, 1000)
    diffuser_output = ddim_diffuser.step(model_output, timestep, sample)
    assert diffuser_output['pre_sample'].shape == (1, 3, 64, 64)