# model settings
model = dict(
    type='CycleGAN',
    generator=dict(
        type='ResnetGenerator',
        in_channels=3,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02)),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    cycle_loss=dict(type='L1Loss', loss_weight=10.0, reduction='mean'),
    id_loss=dict(type='L1Loss', loss_weight=0.5, reduction='mean'))
# model training and testing settings
train_cfg = dict(direction='b2a', buffer_size=50)  # model default: a2b
test_cfg = dict(direction='b2a', show_input=True)

# dataset settings
train_dataset_type = 'GenerationUnpairedDataset'
val_dataset_type = 'GenerationUnpairedDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_a',
        flag='color'),
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_b',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=['img_a', 'img_b'],
        crop_size=(256, 256),
        random_crop=True),
    dict(type='Flip', keys=['img_a'], direction='horizontal'),
    dict(type='Flip', keys=['img_b'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize', keys=['img_a', 'img_b'], to_rgb=True,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_a',
        flag='color'),
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_b',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize', keys=['img_a', 'img_b'], to_rgb=True,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
data_root = 'data/unpaired/facades'
data = dict(
    samples_per_gpu=1,
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
    generators=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=40000, interval=400)

# checkpoint saving
checkpoint_config = dict(interval=4000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=4000, save_image=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])
visual_config = None

# runtime settings
total_iters = 80000
cudnn_benchmark = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
exp_name = 'cyclegan_facades'
work_dir = f'./work_dirs/{exp_name}'
