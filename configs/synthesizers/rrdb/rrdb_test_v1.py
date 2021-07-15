# model settings
from sys import float_repr_style

model = dict(
    type='RRDBnet',
    bgan_generator=dict(
        type='DenseGeneratorFromRRDB',
        in_channels=7,
        out_channels=3,
        base_channels=32,
        block=dict(type='RrdbBlock',
                   nums_block = 3
                   ),
        num_blocks=9,
        use_dropout=False,
        is_skip=True,
        with_nosie=True,
        init_cfg=dict(type='normal', gain=0.02)
        ),
    bgan_discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
    dbgan_generator=dict(
        type='DenseGeneratorFromRRDB',
        in_channels=3,
        out_channels=3,
        base_channels=32,
        block=dict(type='RrdbBlock',
                   nums_block = 3
                   ),
        num_blocks=16,
        use_dropout=False,
        is_skip=True,
        with_nosie=False,
        init_cfg=dict(type='normal', gain=0.02),
        ),
    dbgan_discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
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
    content_loss=dict(type='L1Loss',loss_weight=100.0, reduction='mean'),
    pretrained=None,
    is_useRBL=True
)
# model training and testing settings
test_cfg = dict(metrics=['psnr', 'ssim'])
train_cfg =dict(disc_steps=1)
# dataset settings
train_dataset_type = 'GenerationUnpairedBlurSharpDataset'
val_dataset_type = 'GenerationPairedBlurSharpDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline=[
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_sharp_real',
        flag='color',
        save_original_img='True'),
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_blur_real',
        flag='color',
        save_original_img='True'),
    dict(
        type='Crop',
        keys=['img_sharp_real', 'img_blur_real'],
        crop_size=(128, 128),
        random_crop=True),
    dict(type='Flip', keys=['img_sharp_real'], direction='horizontal'),
    dict(type='Flip', keys=['img_blur_real'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['img_sharp_real', 'img_blur_real']),
    dict(
        type='Normalize', keys=['img_sharp_real', 'img_blur_real'], to_rgb=True,
        **img_norm_cfg),
    dict(type='AddNoiseMap',keys=['img_sharp_real'],generate_type='Gausssion'),
    dict(type='ImageToTensor', keys=['img_sharp_real', 'img_blur_real','noise_map']),
    dict(
        type='Collect',
        keys=['img_sharp_real', 'img_blur_real','noise_map',
        'ori_img_blur_real','ori_img_sharp_real'],
        meta_keys=['img_sharp_real_path', 'img_blur_real_path'])
]
test_pipeline=[
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_sharp_real',
        flag='color',
        save_original_img='True'),
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_blur_real',
        flag='color',
        save_original_img='True'),
    dict(type='RescaleToZeroOne', keys=['img_sharp_real', 'img_blur_real']),
    dict(
        type='Normalize', keys=['img_sharp_real', 'img_blur_real'], to_rgb=True,
        **img_norm_cfg),
    dict(type='AddNoiseMap',keys=['img_sharp_real'],generate_type='Gausssion'),
    dict(type='ImageToTensor', keys=['img_sharp_real', 'img_blur_real','noise_map']),
    dict(
        type='Collect',
        keys=['img_sharp_real', 'img_blur_real','noise_map',
        'ori_img_blur_real','ori_img_sharp_real'],
        meta_keys=['img_sharp_real_path', 'img_blur_real_path'])
]

data_root = '/home/featurize/data/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=train_dataset_type,
        dataroot=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True))

# optimizer
optimizers = dict(
    bgan_generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    bgan_discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    dbgan_generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    dbgan_discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=10960, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=1000, save_image=1)
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])
visual_config = None

# runtime settings
total_iters = 219200
cudnn_benchmark = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
exp_name = 'testDb'
work_dir = f'/home/featurize/work/{exp_name}'