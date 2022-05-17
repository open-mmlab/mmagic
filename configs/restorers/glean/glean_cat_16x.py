exp_name = 'glean_cat_16x'

scale = 16
# model settings
model = dict(
    type='GLEAN',
    generator=dict(
        type='GLEANStyleGANv2',
        in_size=16,
        out_size=256,
        style_channels=512,
        pretrained=dict(
            ckpt_path='http://download.openmmlab.com/mmgen/stylegan2/'
            'official_weights/stylegan2-cat-config-f-official_20210327'
            '_172444-15bc485b.pth',
            prefix='generator_ema')),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=256,
        pretrained=dict(
            ckpt_path='http://download.openmmlab.com/mmgen/stylegan2/'
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
    pretrained=None,
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'SRAnnotationDataset'
val_dataset_type = 'SRAnnotationDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', io_backend='disk', key='lq'),
    dict(type='LoadImageFromFile', io_backend='disk', key='gt'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        to_rgb=True),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
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

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=8, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/cat_train/BIx16_down',
            gt_folder='data/cat_train/GT',
            ann_file='data/cat_train/meta_info_LSUNcat_GT.txt',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/cat_test/BIx16_down',
        gt_folder='data/cat_test/GT',
        ann_file='data/cat_test/meta_info_Cat100_GT.txt',
        pipeline=test_pipeline,
        scale=scale),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/cat_test/BIx16_down',
        gt_folder='data/cat_test/GT',
        ann_file='data/cat_test/meta_info_Cat100_GT.txt',
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
