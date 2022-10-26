"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_8xb4-800kiters_ffhq-256x256.py']

model = dict(loss_config=dict(r1_use_apex_amp=False, g_reg_use_apex_amp=False))

train_cfg = dict(max_iters=800002)

# remain to be refactored
apex_amp = dict(mode='gan', init_args=dict(opt_level='O1', num_losses=2))
resume_from = None
