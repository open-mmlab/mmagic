_base_ = [
    '../_base_/gen_default_runtime.py',
    '../_base_/models/base_styleganv2.py',
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
        momentum=1. - (0.5**(32. / (ema_half_life * 1000.)))),
    loss_config=dict(
        r1_loss_weight=10. / 2. * d_reg_interval,
        r1_interval=d_reg_interval,
        norm_mode='HWC',
        g_reg_interval=g_reg_interval,
        g_reg_weight=2. * g_reg_interval,
        pl_batch_shrink=2))

train_cfg = dict(max_iters=1800002)

optim_wrapper = dict(
    generator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * g_reg_ratio, betas=(0,
                                                        0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * d_reg_ratio, betas=(0,
                                                        0.99**d_reg_ratio))))
# DATA
batch_size = 4
data_root = './data/lsun/images/car'
dataset_type = 'BasicImageDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(
        type='NumpyPad',
        keys='img',
        padding=((64, 64), (0, 0), (0, 0)),
    ),
    dict(type='Flip', keys=['gt'], direction='horizontal'),
    dict(type='PackInputs')
]

val_pipeline = train_pipeline

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

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
        prefix='FID-50k',
        fake_nums=50000,
        real_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K'),
    dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
]
# NOTE: config for save multi best checkpoints
# default_hooks = dict(
#     checkpoint=dict(
#         save_best=['FID-Full-50k/fid', 'IS-50k/is'],
#         rule=['less', 'greater']))
default_hooks = dict(checkpoint=dict(save_best='FID-50k/fid'))

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
