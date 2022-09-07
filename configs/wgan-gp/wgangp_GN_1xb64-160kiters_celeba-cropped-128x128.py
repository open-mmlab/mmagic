_base_ = [
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/models/base_wgan.py',
    '../_base_/gen_default_runtime.py',
]

# `batch_size` and `data_root` need to be set.
batch_size = 4
data_root = './data/celeba-cropped/cropped_images_aligned_png/'
train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

train_cfg = dict(max_iters=160000)

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.9))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.9))))

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=1000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

# METRICS
metrics = [
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
