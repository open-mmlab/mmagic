_base_ = [
    '../_base_/models/dcgan/base_dcgan_128x128.py',
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/gen_default_runtime.py'
]

model = dict(discriminator=dict(output_scale=4, out_channels=1))

# define dataset
batch_size = 64
data_root = './data/celeba-cropped/cropped_images_aligned_png/'
train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99))))

train_cfg = dict(max_iters=160000)

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=20, save_best='FID-Full-50k/fid', rule='less'))

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
        sample_model='orig'),
    dict(
        type='MS_SSIM', prefix='ms-ssim', fake_nums=10000,
        sample_model='orig'),
    dict(
        type='SWD',
        prefix='swd',
        fake_nums=16384,
        sample_model='orig',
        image_shape=(3, 128, 128))
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
