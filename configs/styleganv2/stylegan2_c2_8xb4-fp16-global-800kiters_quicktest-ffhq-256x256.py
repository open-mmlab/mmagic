"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_8xb4-800kiters_ffhq-256x256.py']

model = dict(
    generator=dict(out_size=256, fp16_enabled=True),
    discriminator=dict(in_size=256, fp16_enabled=True),
    disc_auxiliary_loss=dict(data_info=dict(loss_scaler='loss_scaler')),
    # gen_auxiliary_loss=dict(data_info=dict(loss_scaler='loss_scaler')),
)

batch_size = 2
dataset_type = 'QuickTestImageDataset'

train_dataloader = dict(batch_size=batch_size, dataset=dict(type=dataset_type))

val_dataloader = dict(batch_size=batch_size, dataset=dict(type=dataset_type))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(dataset_type=dataset_type))

default_hooks = dict(logger=dict(type='LoggerHook', interval=1))

train_cfg = dict(max_iters=800002)

optim_wrapper = dict(
    generator=dict(type='AmpOptimWrapper', loss_scale=512),
    discriminator=dict(type='AmpOptimWrapper', loss_scale=512))

# METRICS
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
