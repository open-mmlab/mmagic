_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/denoising-gaussian_gray_test_config.py'
]

experiment_name = 'swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# modify sigma of RandomNoise
sigma = 15
test_dataloader = _base_.test_dataloader
for dataloader in test_dataloader:
    test_pipeline = dataloader['dataset']['pipeline']
    test_pipeline[2]['params']['gaussian_sigma'] = [sigma, sigma]

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SwinIRNet',
        upscale=1,
        in_chans=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='',
        resi_connection='1conv'),
    pixel_loss=dict(type='CharbonnierLoss', eps=1e-9),
    data_preprocessor=dict(type='DataPreprocessor', mean=[0.], std=[255.]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[sigma, sigma],
            gaussian_gray_noise_prob=0),
        keys=['img']),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='grayscale',
        imdecode_backend='cv2'),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[sigma, sigma],
            gaussian_gray_noise_prob=0),
        keys=['img']),
    dict(type='PackInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=4,
    batch_size=1,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DFWB8550sub_GT.txt',
        metainfo=dict(dataset_type='dfwb', task_name='denoising'),
        data_root=data_root + '/DFWB',
        data_prefix=dict(img='', gt=''),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='set12', task_name='denoising'),
        data_root=data_root + '/Set12',
        data_prefix=dict(img='', gt=''),
        pipeline=val_pipeline))

val_evaluator = [
    dict(type='PSNR', prefix='Set12'),
    dict(type='SSIM', prefix='Set12'),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1_600_000, val_interval=5000)
val_cfg = dict(type='ValLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    milestones=[800000, 1200000, 1400000, 1500000, 1600000],
    gamma=0.5)
