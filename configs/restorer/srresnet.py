exp_name = '001_G1_MSRResNetX4_DIV2K_fromscratch'

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='MSRResNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=4),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR'], crop_border=4)

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
    # train
    samples_per_gpu=16,
    workers_per_gpu=6,
    drop_last=True,
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            ann_file='./data/DIV2K800_sub.txt',
            data_prefix='./data',
            pipeline=train_pipeline,
            scale=4)),
    # val
    val_samples_per_gpu=1,
    val_workers_per_gpu=1,
    val=dict(
        type=val_dataset_type,
        lq_folder='./data/val_set5/Set5_bicLRx4',
        gt_folder='./data/val_set5/Set5',
        pipeline=test_pipeline,
        scale=4,
        filename_tmpl='{}'),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='./data/DIV2K_valid_LR_bicubic',
        gt_folder='./data/DIV2K_valid_HR',
        pipeline=test_pipeline,
        scale=4,
        filename_tmpl='{}x4'),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 1000000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    period=[250000, 250000, 250000, 250000],
    restarts=[250000, 500000, 750000],
    restart_weights=[1, 1, 1],
    eta_min=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='IterTextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr')),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl', port=29747)
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
