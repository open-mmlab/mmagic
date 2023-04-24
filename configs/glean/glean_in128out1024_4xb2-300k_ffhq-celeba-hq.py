_base_ = '../_base_/models/base_glean.py'

experiment_name = 'glean_in128out1024_4xb2-300k_ffhq-celeba-hq'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

scale = 8
# model settings
model = dict(
    type='SRGAN',
    generator=dict(
        type='GLEANStyleGANv2',
        in_size=128,
        out_size=1024,
        style_channels=512,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://download.openmmlab.com/mmediting/stylegan2/'
            'official_weights/stylegan2-ffhq-config-f-official_20210327'
            '_171224-bce9310c.pth',
            prefix='generator_ema')),
    discriminator=dict(
        type='StyleGANv2Discriminator',
        in_size=1024,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://download.openmmlab.com/mmediting/stylegan2/'
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
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    ),
)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='CopyValues', src_keys=['gt'], dst_keys=['img']),
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
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0, 1, 0],  # up, down, keep
            resize_scale=[0.03125, 1],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['img'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[0, 50],
            gaussian_gray_noise_prob=0),
        keys=['img'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[5, 50]),
        keys=['img']),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(1024, 1024),
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['img'],
    ),
    dict(type='Clip', keys=['img']),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(128, 128), resize_opt=['area'], resize_prob=[1]),
        keys=['img'],
    ),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(128, 128), resize_opt=['area'], resize_prob=[1]),
        keys=['img'],
    ),
    dict(type='PackInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=6,
    batch_size=2,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='ffhq_celebahq', task_name='sisr'),
        data_root='data/FFHQ_CelebAHQ',
        data_prefix=dict(gt='GT'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='ffhq', task_name='sisr'),
        data_root='data/CelebA-HQ',
        data_prefix=dict(img='BIx8_down', gt='GT'),
        ann_file='meta_info_CelebAHQ_val100_GT.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader
