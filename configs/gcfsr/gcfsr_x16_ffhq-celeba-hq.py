"""Config for the `config` setting in GCFSR."""

_base_ = ['../_base_/datasets/ffhq_flip.py', '../_base_/models/base_gcfsr.py']

# reg params
d_reg_interval = 16
g_reg_interval = 4

d_reg_ratio = d_reg_interval / (d_reg_interval + 1)
g_reg_ratio = g_reg_interval / (g_reg_interval + 1)

ema_half_life = 10.  # G_smoothing_kimg

# model
model = dict(
    type='GCFSRGAN',
    generator=dict(
        type='GCFSR',
        out_size=1024,
        num_style_feat=512,
    ),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=1024,
    ),
    ema_config=dict(
        type='ExponentialMovingAverage',
        interval=1,
        momentum=0.5**(32. / (ema_half_life * 1000.))),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'21': 1.0},
        vgg_type='vgg16',
        perceptual_weight=1e-2,
        style_weight=0,
        norm_img=True,
        criterion='l1',
        pretrained='torchvision://vgg16'),
    gan_loss=dict(type='GANLoss', gan_type='wgan_softplus', loss_weight=1e-2),
    data_preprocessor=dict(type='GenDataPreprocessor'),
    rescale_list=[64, 64, 64, 64, 32, 32, 16, 16, 8, 4],
    d_reg_interval=d_reg_interval,
    r1_reg_weight=10,
)

optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(
            type='Adam', lr=2e-3 * g_reg_ratio, betas=(0, 0.99**g_reg_ratio))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(
            type='Adam', lr=2e-3 * d_reg_ratio, betas=(0, 0.99**d_reg_ratio))),
)

batch_size = 1
num_workers = 2 * batch_size
data_root = './data/ffhq/ffhq_images/'
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(data_root=data_root))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(data_root=data_root))
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(data_root=data_root))
