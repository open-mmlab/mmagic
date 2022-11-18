_base_ = [
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/gen_default_runtime.py',
]

# MODEL
loss_config = dict(gp_norm_mode='HWC', gp_loss_weight=10)
model = dict(
    type='WGANGP',
    data_preprocessor=dict(type='GenDataPreprocessor'),
    generator=dict(type='WGANGPGenerator', noise_size=128, out_scale=128),
    discriminator=dict(
        type='WGANGPDiscriminator',
        in_channel=3,
        in_scale=128,
        conv_module_cfg=dict(
            conv_cfg=None,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            norm_cfg=dict(type='GN'),
            order=('conv', 'norm', 'act'))),
    discriminator_steps=5,
    loss_config=loss_config)

# `batch_size` and `data_root` need to be set.
batch_size = 64
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
        interval=5000,
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

# save multi best checkpoints
default_hooks = dict(checkpoint=dict(save_best='swd/avg'))

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
