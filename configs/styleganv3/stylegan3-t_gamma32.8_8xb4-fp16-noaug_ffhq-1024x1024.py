_base_ = [
    '../_base_/models/base_styleganv3.py',
    '../_base_/datasets/ffhq_flip.py',
    '../_base_/gen_default_runtime.py',
]

batch_size = 32
magnitude_ema_beta = 0.5**(batch_size / (20 * 1e3))
synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 32768,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}
r1_gamma = 32.8
d_reg_interval = 16

ema_config = dict(
    type='RampUpEMA',
    interval=1,
    ema_kimg=10,
    ema_rampup=0.05,
    batch_size=batch_size,
    eps=1e-8,
    start_iter=0)

model = dict(
    generator=dict(out_size=1024, img_channels=3, synthesis_cfg=synthesis_cfg),
    discriminator=dict(in_size=1024),
    loss_config=dict(r1_loss_weight=r1_gamma / 2.0 * d_reg_interval),
    ema_config=ema_config)

g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

optim_wrapper = dict(
    generator=dict(
        optimizer=dict(
            type='Adam', lr=0.0025 * g_reg_ratio, betas=(0,
                                                         0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * d_reg_ratio, betas=(0,
                                                        0.99**d_reg_ratio))))

batch_size = 4
data_root = 'data/ffhq/images'

train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

train_cfg = dict(max_iters=800002)

# VIS_HOOK
custom_hooks = [
    dict(
        type='VisualizationHook',
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
        sample_model='ema')
]
# NOTE: config for save multi best checkpoints
# default_hooks = dict(
#     checkpoint=dict(
#         save_best=['FID-Full-50k/fid', 'IS-50k/is'],
#         rule=['less', 'greater']))
default_hooks = dict(checkpoint=dict(save_best='FID-Full-50k/fid'))

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
