"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_8xb4-800kiters_ffhq-256x256.py']

model = dict(
    generator=dict(out_size=256, num_fp16_scales=4),
    discriminator=dict(in_size=256, num_fp16_scales=4),
    loss_config=dict(scale_r1_loss=True))

optim_wrapper = dict(
    generator=dict(type='AmpOptimWrapper', loss_scale=512),
    discriminator=dict(type='AmpOptimWrapper', loss_scale=512))
