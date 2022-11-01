_base_ = [
    '../_base_/models/dcgan/base_dcgan_64x64.py',
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/gen_default_runtime.py'
]

# output single channel
model = dict(generator=dict(out_channels=1), discriminator=dict(in_channels=1))

# define dataset
# modify train_pipeline to load gray scale images
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        io_backend='disk',
        color_type='grayscale'),
    dict(type='Resize', scale=(64, 64)),
    dict(type='PackEditInputs', meta_keys=[])
]

# set ``batch_size``` and ``data_root```
batch_size = 128
data_root = 'data/mnist_64/train'
train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

default_hooks = dict(
    checkpoint=dict(
        interval=500,
        save_best=['swd/avg', 'ms-ssim/avg'],
        rule=['less', 'greater']))

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=10000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

train_cfg = dict(max_iters=5000)

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
        image_shape=(3, 64, 64))
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0004, betas=(0.5, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999))))
