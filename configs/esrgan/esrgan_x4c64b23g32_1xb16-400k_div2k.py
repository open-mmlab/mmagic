_base_ = './esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py'

experiment_name = 'esrgan_x4c64b23g32_1xb16-400k_div2k'
work_dir = f'./work_dirs/{experiment_name}'

scale = 4

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
pretrain_generator_url = (
    'https://download.openmmlab.com/mmediting/restorers/esrgan'
    '/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth')
model = dict(
    type='ESRGAN',
    generator=dict(
        type='RRDBNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=23,
        growth_channels=32,
        upscale_factor=scale,
        init_cfg=dict(type='Pretrained', checkpoint=pretrain_generator_url)),
    discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
    pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-3,
        real_label_val=1.0,
        fake_label_val=0),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=400_000,
    val_interval=5000)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))),
)

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=False,
    milestones=[50000, 100000, 200000, 300000],
    gamma=0.5)
