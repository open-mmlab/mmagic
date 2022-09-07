"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_8xb4-800kiters_ffhq-256x256.py']

model = dict(
    generator=dict(out_size=256),
    discriminator=dict(in_size=256, convert_input_fp32=False),
)

# remain to be refactored
apex_amp = dict(
    mode='gan', init_args=dict(opt_level='O1', num_losses=2, loss_scale=512.))

train_cfg = dict(max_iters=800002)

batch_size = 2
dataset_type = 'QuickTestImageDataset'

train_dataloader = dict(batch_size=batch_size, dataset=dict(type=dataset_type))

val_dataloader = dict(batch_size=batch_size, dataset=dict(type=dataset_type))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(dataset_type=dataset_type))

default_hooks = dict(logger=dict(type='LoggerHook', interval=1))

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
