exp_name = 'srcnn_x4k915_g1_1000k_div2k'

scale = 4
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='SRCNN',
        channels=(3, 64, 32, 3),
        kernel_sizes=(9, 1, 5),
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRAnnotationDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'lq_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X4_sub',
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',
            ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/val_set5/Set5_bicLRx4',
        gt_folder='data/val_set5/Set5',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/val_set5/Set5_bicLRx4',
        gt_folder='data/val_set5/Set5',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 1000000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[250000, 250000, 250000, 250000],
    restart_weights=[1, 1, 1, 1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
