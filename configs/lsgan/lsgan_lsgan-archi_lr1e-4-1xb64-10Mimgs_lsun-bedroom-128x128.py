_base_ = [
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/gen_default_runtime.py'
]
# define model
model = dict(
    type='LSGAN',
    noise_size=1024,
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(
        type='LSGANGenerator',
        output_scale=128,
        base_channels=256,
        noise_size=1024),
    discriminator=dict(
        type='LSGANDiscriminator', input_scale=128, base_channels=64))

total_iters = 160000
disc_step = 1
train_cfg = dict(max_iters=total_iters * disc_step)

# define dataset
batch_size = 64
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
# adjust running config
# METRICS
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-50k',
        fake_nums=50000,
        real_nums=50000,
        inception_style='PyTorch',
        sample_model='orig')
]
val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
