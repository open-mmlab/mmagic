exp_name = 'ttsr-gan_x4_c64b16_g1_500k_CUFED'
scale = 4

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
        fake_label_val=0))
# model training and testing settings
train_cfg = dict(fix_iter=25000, disc_steps=2)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRFolderRefDataset'
val_dataset_type = 'SRFolderRefDataset'
test_dataset_type = 'SRFolderRefDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='ref',
        flag='color',
        channel_order='rgb',
        backend='pillow'),
    dict(type='CropLike', target_key='ref', reference_key='gt'),
    dict(
        type='Resize',
        scale=1 / scale,
        keep_ratio=True,
        keys=['gt', 'ref'],
        output_keys=['lq', 'ref_down'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=float(scale),
        keep_ratio=True,
        keys=['lq', 'ref_down'],
        output_keys=['lq_up', 'ref_downup'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]),
    dict(
        type='Normalize',
        keys=['lq_up', 'ref', 'ref_downup'],
        mean=[0., 0., 0.],
        std=[255., 255., 255.]),
    dict(
        type='Flip',
        keys=['lq', 'gt', 'lq_up'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip',
        keys=['lq', 'gt', 'lq_up'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(
        type='RandomTransposeHW',
        keys=['lq', 'gt', 'lq_up'],
        transpose_ratio=0.5),
    dict(
        type='Flip',
        keys=['ref', 'ref_downup'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip',
        keys=['ref', 'ref_downup'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(
        type='RandomTransposeHW',
        keys=['ref', 'ref_downup'],
        transpose_ratio=0.5),
    dict(
        type='ImageToTensor', keys=['lq', 'gt', 'lq_up', 'ref', 'ref_downup']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'lq_up', 'ref', 'ref_downup'],
        meta_keys=['gt_path', 'ref_path'])
]
valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='ref',
        flag='color',
        channel_order='rgb',
        backend='pillow'),
    dict(type='CropLike', target_key='ref', reference_key='gt'),
    dict(
        type='Resize',
        scale=1 / scale,
        keep_ratio=True,
        keys=['gt', 'ref'],
        output_keys=['lq', 'ref_down'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=float(scale),
        keep_ratio=True,
        keys=['lq', 'ref_down'],
        output_keys=['lq_up', 'ref_downup'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]),
    dict(
        type='Normalize',
        keys=['lq_up', 'ref', 'ref_downup'],
        mean=[0., 0., 0.],
        std=[255., 255., 255.]),
    dict(
        type='ImageToTensor', keys=['lq', 'gt', 'lq_up', 'ref', 'ref_downup']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'lq_up', 'ref', 'ref_downup'],
        meta_keys=['gt_path', 'ref_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='ref',
        flag='color',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='Resize',
        scale=1 / scale,
        keep_ratio=True,
        keys=['ref'],
        output_keys=['ref_down'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=float(scale),
        keep_ratio=True,
        keys=['lq', 'ref_down'],
        output_keys=['lq_up', 'ref_downup'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]),
    dict(
        type='Normalize',
        keys=['lq_up', 'ref', 'ref_downup'],
        mean=[0., 0., 0.],
        std=[255., 255., 255.]),
    dict(type='ImageToTensor', keys=['lq', 'lq_up', 'ref', 'ref_downup']),
    dict(
        type='Collect',
        keys=['lq', 'lq_up', 'ref', 'ref_downup'],
        meta_keys=['lq_path', 'ref_path'])
]

data = dict(
    workers_per_gpu=9,
    train_dataloader=dict(samples_per_gpu=9, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=52,
        dataset=dict(
            type=train_dataset_type,
            gt_folder='data/CUFED/train/input/',
            ref_folder='data/CUFED/train/ref/',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        gt_folder='data/CUFED/valid/input_format/',
        ref_folder='data/CUFED/valid/ref1_format/',
        pipeline=valid_pipeline,
        scale=scale),
    test=dict(
        type=test_dataset_type,
        gt_folder='data/CUFED/valid/input_format/',
        ref_folder='data/CUFED/valid/ref1_format/',
        pipeline=valid_pipeline,
        scale=scale))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 500000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[100000, 200000, 300000, 400000],
    gamma=0.5)

checkpoint_config = dict(interval=100, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
