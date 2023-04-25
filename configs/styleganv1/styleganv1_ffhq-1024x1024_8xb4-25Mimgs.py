_base_ = [
    '../_base_/models/base_styleganv1.py',
    '../_base_/datasets/grow_scale_imgs_ffhq_styleganv1.py',
    '../_base_/gen_default_runtime.py',
]

# MODEL
model_wrapper_cfg = dict(find_unused_parameters=True)
ema_half_life = 10.  # G_smoothing_kimg
ema_config = dict(
    interval=1, momentum=1. - (0.5**(32. / (ema_half_life * 1000.))))
model = dict(
    generator=dict(out_size=1024),
    discriminator=dict(in_size=1024),
    nkimgs_per_scale={
        '8': 1200,
        '16': 1200,
        '32': 1200,
        '64': 1200,
        '128': 1200,
        '256': 1200,
        '512': 1200,
        '1024': 166000
    },
    ema_config=ema_config)

# TRAIN
train_cfg = dict(max_iters=670000)

optim_wrapper = dict(
    constructor='PGGANOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    lr_schedule=dict(
        generator={
            '128': 0.0015,
            '256': 0.002,
            '512': 0.003,
            '1024': 0.003
        },
        discriminator={
            '128': 0.0015,
            '256': 0.002,
            '512': 0.003,
            '1024': 0.003
        }))

# VIS_HOOK + DATAFETCH
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img')),
    dict(type='PGGANFetchDataHook')
]

# METRICS
inception_pkl = './work_dirs/ffhq1024-full.pkl'
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        inception_pkl=inception_pkl,
        sample_model='ema'),
    dict(type='PrecisionAndRecall', fake_nums=50000, k=3, prefix='PR-50K'),
]
default_hooks = dict(checkpoint=dict(save_best='FID-Full-50k/fid'))

val_evaluator = test_evaluator = dict(metrics=metrics)
