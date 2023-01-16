_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/sisr_x4_test_config.py'
]

experiment_name = 'swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4
img_size = 48

# evaluated on Y channels
test_evaluator = _base_.test_evaluator
for evaluator in test_evaluator:
    for metric in evaluator:
        metric['convert_to'] = 'Y'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SwinIRNet',
        upscale=scale,
        in_chans=3,
        img_size=img_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='EditDataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=img_size * scale),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=4,
    batch_size=4,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DIV2K800sub_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',
        data_prefix=dict(
            img='DIV2K_train_LR_bicubic/X4_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',
        data_prefix=dict(img='LRbicx4', gt='GTmod12'),
        pipeline=val_pipeline))

val_evaluator = [
    dict(type='PSNR', crop_border=scale),
    dict(type='SSIM', crop_border=scale),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=500_000, val_interval=5000)
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
    milestones=[250000, 400000, 450000, 475000],
    gamma=0.5)
