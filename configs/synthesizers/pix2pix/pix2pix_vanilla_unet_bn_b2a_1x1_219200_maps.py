# model settings
model = dict(
    type='Pix2Pix',
    generator=dict(
        type='UnetGenerator',
        in_channels=3,
        out_channels=3,
        num_down=8,
        base_channels=64,
        norm_cfg=dict(type='BN'),
        use_dropout=True,
        init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=6,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='normal', gain=0.02)),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    pixel_loss=dict(type='L1Loss', loss_weight=100.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(direction='b2a')  # model default: a2b
test_cfg = dict(direction='b2a', show_input=True)

# dataset settings
train_dataset_type = 'GenerationPairedDataset'
val_dataset_type = 'GenerationPairedDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(type='FixedCrop', keys=['img_a', 'img_b'], crop_size=(256, 256)),
    dict(type='Flip', keys=['img_a', 'img_b'], direction='horizontal'),
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
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
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
data_root = './data/paired/maps'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
    val_samples_per_gpu=1,
    val_workers_per_gpu=0,
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
    generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=10960, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=21920, save_image=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
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
exp_name = 'pix2pix_maps_b2a'
work_dir = f'./work_dirs/{exp_name}'
