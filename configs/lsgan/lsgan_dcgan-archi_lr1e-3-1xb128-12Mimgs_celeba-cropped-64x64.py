_base_ = [
    '../_base_/models/dcgan/base_dcgan_64x64.py',
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/gen_default_runtime.py'
]
model = dict(type='LSGAN')
total_iters = 100000
disc_step = 1
train_cfg = dict(max_iters=total_iters * disc_step)
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
    generator=dict(optimizer=dict(type='Adam', lr=0.001, betas=(0.5, 0.99))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.001, betas=(0.5, 0.99))))

# VIS_HOOK
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
default_hooks = dict(
    checkpoint=dict(
        save_best=['FID-Full-50k/fid', 'IS-50k/is'], rule=['less', 'greater']))

# METRICS
metrics = [
    dict(
        type='InceptionScore',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig'),
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-50k',
        real_nums=50000,
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
