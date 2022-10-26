_base_ = [
    '../_base_/datasets/imagenet_noaug_128.py',
    '../_base_/gen_default_runtime.py',
]

# setting image size to 512x512
train_resize = _base_.train_dataloader.dataset.pipeline[3]
test_resize = _base_.test_dataloader.dataset.pipeline[3]
val_resize = _base_.val_dataloader.dataset.pipeline[3]
train_resize.scale = test_resize.scale = val_resize.scale = (256, 256)

ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.9999,
    update_buffers=True,
    start_iter=20000)

model = dict(
    type='BigGAN',
    num_classes=1000,
    data_preprocessor=dict(type='GenDataPreprocessor'),
    ema_config=ema_config,
    generator=dict(
        type='BigGANDeepGenerator',
        output_scale=256,
        noise_size=128,
        num_classes=1000,
        base_channels=128,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        sn_style='torch',
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        concat_noise=True,
        auto_sync_bn=False,
        rgb2bgr=True),
    discriminator=dict(
        type='BigGANDeepDiscriminator',
        input_scale=256,
        num_classes=1000,
        base_channels=128,
        sn_eps=1e-6,
        sn_style='torch',
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True))

train_cfg = train_dataloader = optim_wrapper = None

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
val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
