_base_ = [
    '../_base_/models/dcgan/base_dcgan_128x128.py',
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/gen_default_runtime.py'
]
model = dict(type='LSGAN', discriminator=dict(output_scale=4, out_channels=1))

total_iters = 160000
disc_step = 1
train_cfg = dict(max_iters=total_iters * disc_step)

# define dataset
# you must set `samples_per_gpu` and `imgs_root`
# `batch_size` and `data_root` need to be set.
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
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)

# TODO
# metrics = dict(
#     ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
#     swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 128, 128)),
#     fid50k=dict(type='FID', num_images=50000, inception_pkl=None))
