_base_ = [
    '../_base_/datasets/ffhq_flip.py', '../_base_/models/base_styleganv2.py',
    '../_base_/gen_default_runtime.py'
]

ema_half_life = 10.
ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.5**(32. / (ema_half_life * 1000.)))

model = dict(
    type='MSPIEStyleGAN2',
    generator=dict(
        type='MSStyleGANv2Generator',
        head_pos_encoding=dict(type='CSG'),
        deconv2conv=True,
        up_after_conv=True,
        head_pos_size=(4, 4),
        up_config=dict(scale_factor=2, mode='bilinear', align_corners=True),
        out_size=256),
    discriminator=dict(
        type='MSStyleGAN2Discriminator', in_size=256, with_adaptive_pool=True),
    train_settings=dict(
        num_upblocks=6,
        multi_input_scales=[0, 2, 4],
        multi_scale_probability=[0.5, 0.25, 0.25]),
    ema_config=ema_config)

train_cfg = dict(max_iters=1100002)

# `batch_size` and `data_root` need to be set.
batch_size = 3
data_root = './data/ffhq/ffhq_imgs/ffhq_512'
train_dataloader = dict(
    batch_size=batch_size,  # set by user
    dataset=dict(data_root=data_root))

pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Resize', scale=(256, 256)),
    dict(type='PackEditInputs', keys=['img'])
]

val_dataloader = dict(
    batch_size=batch_size,  # set by user
    dataset=dict(data_root=data_root, pipeline=pipeline))

test_dataloader = dict(
    batch_size=batch_size,  # set by user
    dataset=dict(data_root=data_root, pipeline=pipeline))

# define optimizer
d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

optim_wrapper = dict(
    generator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * g_reg_ratio, betas=(0,
                                                        0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * d_reg_ratio, betas=(0,
                                                        0.99**d_reg_ratio))))

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
default_hooks = dict(checkpoint=dict(save_best=['FID-50k/fid'], rule=['less']))
# METRICS
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-50k',
        fake_nums=50000,
        real_nums=50000,
        inception_style='pytorch',
        sample_model='ema'),
    dict(type='PrecisionAndRecall', fake_nums=10000, prefix='PR-10K')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
