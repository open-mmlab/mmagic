_base_ = [
    '../_base_/models/dcgan/base_dcgan_64x64.py',
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/gen_default_runtime.py'
]

# define dataset
# batch_size and data_root must be set
batch_size = 128
data_root = './data/celeba-cropped/cropped_images_aligned_png/'
train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))))

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=10000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

model = dict(type='DCGAN')
train_cfg = dict(max_iters=300002)

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
# save best checkpoints
default_hooks = dict(checkpoint=dict(save_best='swd/avg', rule='less'))

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
