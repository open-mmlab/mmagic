_base_ = '../_base_/models/base_glean.py'

experiment_name = 'glean_x8_2xb8_cat'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

scale = 8
# model settings
model = dict(
    type='SRGAN',
    generator=dict(
        type='GLEANStyleGANv2',
        in_size=32,
        out_size=256,
        style_channels=512,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://download.openmmlab.com/mmediting/stylegan2/'
            'official_weights/stylegan2-cat-config-f-official_20210327'
            '_172444-15bc485b.pth',
            prefix='generator_ema')),
    discriminator=dict(
        type='StyleGANv2Discriminator',
        in_size=256,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://download.openmmlab.com/mmediting/stylegan2/'
            'official_weights/stylegan2-cat-config-f-official_20210327'
            '_172444-15bc485b.pth',
            prefix='discriminator')),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'21': 1.0},
        vgg_type='vgg16',
        perceptual_weight=1e-2,
        style_weight=0,
        norm_img=False,
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
        type='EditDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    ),
)

train_pipeline = [
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
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(type='PackEditInputs')
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
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=8,
    batch_size=8,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='cat', task_name='sisr'),
        data_root='data/cat_train',
        data_prefix=dict(img='BIx8_down', gt='GT'),
        ann_file='meta_info_LSUNcat_GT.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='cat', task_name='sisr'),
        data_root='data/cat_test',
        data_prefix=dict(img='BIx8_down', gt='GT'),
        ann_file='meta_info_Cat100_GT.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader
