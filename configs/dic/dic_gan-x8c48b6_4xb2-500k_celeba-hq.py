_base_ = './dic_x8c48b6_4xb2-150k_celeba-hq.py'

experiment_name = 'dic_gan-x8c48b6_4xb2-500k_celeba-hq'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

scale = 8

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
pretrained_light_cnn = 'https://download.openmmlab.com/mmediting/' + \
        'restorers/dic/light_cnn_feature.pth'
model = dict(
    type='DIC',
    generator=dict(
        type='DICNet', in_channels=3, out_channels=3, mid_channels=48),
    discriminator=dict(type='LightCNN', in_channels=3),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean'),
    feature_loss=dict(
        type='LightCNNFeatureLoss',
        pretrained=pretrained_light_cnn,
        loss_weight=0.1,
        criterion='l1'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.005,
        real_label_val=1.0,
        fake_label_val=0),
    train_cfg=dict(pixel_init=10000, disc_repeat=2),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[129.795, 108.12, 96.39],
        std=[255, 255, 255],
    ))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=500_000, val_interval=5000)

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-4)),
    discriminator=dict(
        type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-5)))

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=False,
    milestones=[100000, 200000, 300000, 400000],
    gamma=0.5)
