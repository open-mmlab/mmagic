_base_ = [
    '../_base_/datasets/cifar10_noaug.py',
    '../_base_/gen_default_runtime.py',
]

# define model
ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.9999,
    start_iter=1000)

model = dict(
    type='BigGAN',
    num_classes=10,
    data_preprocessor=dict(type='GenDataPreprocessor', rgb_to_bgr=True),
    generator=dict(
        type='BigGANGenerator',
        output_scale=32,
        noise_size=128,
        num_classes=10,
        base_channels=64,
        with_shared_embedding=False,
        sn_eps=1e-8,
        sn_style='torch',
        init_type='N02',
        split_noise=False,
        auto_sync_bn=False),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=32,
        num_classes=10,
        base_channels=64,
        sn_eps=1e-8,
        sn_style='torch',
        init_type='N02',
        with_spectral_norm=True),
    generator_steps=1,
    discriminator_steps=4,
    ema_config=ema_config)

# define dataset
train_dataloader = dict(batch_size=25, num_workers=8)
val_dataloader = dict(batch_size=25, num_workers=8)
test_dataloader = dict(batch_size=25, num_workers=8)

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        # vis ema and orig at the same time
        vis_kwargs_list=dict(
            type='Noise',
            name='fake_img',
            sample_model='ema/orig',
            target_keys=['ema', 'orig'])),
]

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999))))
train_cfg = dict(max_iters=500000)

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
