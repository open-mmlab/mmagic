_base_ = [
    '../_base_/models/biggan/base_biggan_128x128.py',
    '../_base_/datasets/imagenet_noaug_128.py',
    '../_base_/gen_default_runtime.py',
]

# define model
ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.0001,
    update_buffers=True,
    start_iter=20000)

model = dict(ema_config=ema_config)
train_cfg = dict(max_iters=1500000)

# define dataset
train_dataloader = dict(
    batch_size=32, num_workers=8, dataset=dict(data_root='data/imagenet'))

# define optimizer
optim_wrapper = dict(
    generator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6)),
    discriminator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6)))

# VIS_HOOK
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=10000,
        fixed_input=True,
        # vis ema and orig at the same time
        vis_kwargs_list=dict(
            type='Noise',
            name='fake_img',
            sample_model='ema/orig',
            target_keys=['ema', 'orig'])),
]

metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(
        type='IS',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
# save multi best checkpoints
default_hooks = dict(
    checkpoint=dict(
        save_best=['FID-Full-50k/fid', 'IS-50k/is'], rule=['less', 'greater']))

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
