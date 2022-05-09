exp_name = 'glean_in128out1024_4x2_300k_ffhq_celebahq'

scale = 8
# model settings
model = dict(
    type='GLEAN',
    generator=dict(
        type='GLEANStyleGANv2',
        in_size=128,
        out_size=1024,
        style_channels=512,
        pretrained=dict(
            ckpt_path='http://download.openmmlab.com/mmgen/stylegan2/'
            'official_weights/stylegan2-ffhq-config-f-official_20210327'
            '_171224-bce9310c.pth',
            prefix='generator_ema')),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=1024,
        pretrained=dict(
            ckpt_path='http://download.openmmlab.com/mmgen/stylegan2/'
            'official_weights/stylegan2-ffhq-config-f-official_20210327'
            '_171224-bce9310c.pth',
            prefix='discriminator')),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'21': 1.0},
        vgg_type='vgg16',
        perceptual_weight=1e-2,
        style_weight=0,
        norm_img=True,
        criterion='mse',
        pretrained='torchvision://vgg16'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=1e-2,
        real_label_val=1.0,
        fake_label_val=0),
    pretrained=None,
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRAnnotationDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['gt']),
    dict(type='CopyValues', src_keys=['gt'], dst_keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[41],
            kernel_list=['iso', 'aniso'],
            kernel_prob=[0.5, 0.5],
            sigma_x=[0.2, 10],
            sigma_y=[0.2, 10],
            rotate_angle=[-3.1416, 3.1416],
        ),
        keys=['lq'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0, 1, 0],  # up, down, keep
            resize_scale=[0.03125, 1],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['lq'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[0, 50],
            gaussian_gray_noise_prob=0),
        keys=['lq'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[5, 50]),
        keys=['lq']),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(1024, 1024),
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['lq'],
    ),
    dict(type='Quantize', keys=['lq']),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(128, 128), resize_opt=['area'], resize_prob=[1]),
        keys=['lq'],
    ),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['gt_path'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', io_backend='disk', key='lq'),
    dict(type='LoadImageFromFile', io_backend='disk', key='gt'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

demo_pipeline = [
    dict(
        type='RandomResize',
        params=dict(
            target_size=(128, 128), resize_opt=['area'], resize_prob=[1]),
        keys=['lq'],
    ),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=[])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=30,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/FFHQ_CelebAHQ/GT',
            gt_folder='data/FFHQ_CelebAHQ/GT',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/CelebA-HQ/BIx8_down',
        gt_folder='data/CelebA-HQ/GT',
        ann_file='data/CelebA-HQ/meta_info_CelebAHQ_val100_GT.txt',
        pipeline=test_pipeline,
        scale=scale),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/CelebA-HQ/BIx8_down',
        gt_folder='data/CelebA-HQ/GT',
        ann_file='data/CelebA-HQ/meta_info_CelebAHQ_val100_GT.txt',
        pipeline=test_pipeline,
        scale=scale))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 300000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
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
