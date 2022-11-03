_base_ = './ttsr-rec_x4c64b16_1xb9-200k_CUFED.py'

experiment_name = 'ttsr-gan_x4c64b16_1xb9-500k_CUFED'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'
scale = 4

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
model = dict(
    type='TTSR',
    generator=dict(
        type='TTSRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=(16, 16, 8, 4)),
    extractor=dict(type='LTE'),
    transformer=dict(type='SearchTransformer'),
    discriminator=dict(type='TTSRDiscriminator', in_size=160),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'29': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1e-2,
        style_weight=0,
        criterion='mse'),
    transferal_perceptual_loss=dict(
        type='TransferalPerceptualLoss',
        loss_weight=1e-2,
        use_attention=False,
        criterion='mse'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=1e-3,
        real_label_val=1.0,
        fake_label_val=0),
    train_cfg=dict(pixel_init=25000, disc_repeat=2),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    ))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=500_000, val_interval=5000)

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))),
    extractor=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-5, betas=(0.9, 0.999))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-5, betas=(0.9, 0.999))))

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=False,
    milestones=[100000, 200000, 300000, 400000],
    gamma=0.5)
