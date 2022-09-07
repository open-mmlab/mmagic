"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_8xb4-800kiters_ffhq-256x256.py']

model = dict(
    generator=dict(out_size=256, fp16_enabled=True),
    discriminator=dict(in_size=256, fp16_enabled=False, num_fp16_scales=4),
)
train_cfg = dict(max_iters=800000)
optim_wrapper = dict(
    generator=dict(type='AmpOptimWrapper', loss_scale=512),
    discriminator=dict(type='AmpOptimWrapper', loss_scale=512))
