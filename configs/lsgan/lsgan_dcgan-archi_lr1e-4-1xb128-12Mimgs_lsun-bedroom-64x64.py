_base_ = [
    '../_base_/models/dcgan/base_dcgan_64x64.py',
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/gen_default_runtime.py'
]

model = dict(type='LSGAN', discriminator=dict(output_scale=4, out_channels=1))
total_iters = 100000
disc_step = 1
train_cfg = dict(max_iters=total_iters * disc_step)

# define dataset
# `batch_size` and `data_root` need to be set.
batch_size = 128
data_root = './data/lsun/images/bedroom_train'

train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

test_bs = 32
val_dataloader = dict(batch_size=test_bs, dataset=dict(data_root=data_root))

test_dataloader = dict(batch_size=test_bs, dataset=dict(data_root=data_root))

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99))))
default_hooks = dict(
    checkpoint=dict(save_best=['FID-Full-50k/fid'], rule=['less']))

# METRICS
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig')
]
val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
