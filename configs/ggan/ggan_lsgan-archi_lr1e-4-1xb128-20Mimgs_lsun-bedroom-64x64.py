_base_ = [
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/gen_default_runtime.py'
]

model = dict(
    type='GGAN',
    noise_size=1024,
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(type='LSGANGenerator', output_scale=64),
    discriminator=dict(type='LSGANDiscriminator', input_scale=64))

# define dataset
batch_size = 128
data_root = 'data/lsun/images/bedroom_train'
train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99))))

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=20,
        save_best=['FID-Full-50k/fid', 'swd/avg', 'ms-ssim/avg'],
        rule=['less', 'less', 'greater']))

# VIS_HOOK
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

train_cfg = dict(max_iters=160000)

# METRICS
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig'),
    dict(
        type='MS_SSIM', prefix='ms-ssim', fake_nums=10000,
        sample_model='orig'),
    dict(
        type='SWD',
        prefix='swd',
        fake_nums=16384,
        sample_model='orig',
        image_shape=(3, 64, 64))
]

val_metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig'),
]

val_evaluator = dict(metrics=val_metrics)
test_evaluator = dict(metrics=metrics)
