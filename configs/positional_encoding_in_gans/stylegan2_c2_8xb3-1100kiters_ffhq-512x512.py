"""Config for the `config-f` setting in StyleGAN2."""

_base_ = [
    '../_base_/datasets/ffhq_flip.py', '../_base_/models/base_styleganv2.py',
    '../_base_/gen_default_runtime.py'
]

# reg params
d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

ema_half_life = 10.  # G_smoothing_kimg

model = dict(
    generator=dict(out_size=512),
    discriminator=dict(in_size=512),
    ema_config=dict(
        type='ExponentialMovingAverage',
        interval=1,
        momentum=0.5**(32. / (ema_half_life * 1000.))))

optim_wrapper = dict(
    generator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * g_reg_ratio, betas=(0,
                                                        0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * d_reg_ratio, betas=(0,
                                                        0.99**d_reg_ratio))))

batch_size = 3
data_root = './data/ffhq/ffhq_imgs/ffhq_512'

train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

train_cfg = dict(max_iters=1100002)

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

# METRICS
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(type='PrecisionAndRecall', fake_nums=10000, prefix='PR-10K')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
