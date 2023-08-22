exp_name = 'cain_b5_g1b32_vimeo90k_triplet'

# model settings
model = dict(
    type='CAIN',
    generator=dict(type='CAINNet'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'VFIVimeo90KDataset'
val_dataset_type = 'VFIVimeo90KDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='inputs',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='target',
        channel_order='rgb',
        backend='pillow'),
    dict(type='FixedCrop', keys=['inputs', 'target'], crop_size=(256, 256)),
    dict(
        type='Flip',
        keys=['inputs', 'target'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip',
        keys=['inputs', 'target'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(
        type='ColorJitter',
        keys=['inputs', 'target'],
        channel_order='rgb',
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.05),
    dict(type='TemporalReverse', keys=['inputs'], reverse_ratio=0.5),
    dict(type='RescaleToZeroOne', keys=['inputs', 'target']),
    dict(type='FramesToTensor', keys=['inputs']),
    dict(type='ImageToTensor', keys=['target']),
    dict(
        type='Collect',
        keys=['inputs', 'target'],
        meta_keys=['inputs_path', 'target_path', 'key'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='inputs',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='target',
        channel_order='rgb',
        backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['inputs', 'target']),
    dict(type='FramesToTensor', keys=['inputs']),
    dict(type='ImageToTensor', keys=['target']),
    dict(
        type='Collect',
        keys=['inputs', 'target'],
        meta_keys=['inputs_path', 'target_path', 'key'])
]

demo_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='inputs',
        channel_order='rgb',
        backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['inputs']),
    dict(type='FramesToTensor', keys=['inputs']),
    dict(type='Collect', keys=['inputs'], meta_keys=['inputs_path', 'key'])
]

root_dir = 'data/vimeo_triplet'
data = dict(
    workers_per_gpu=32,
    train_dataloader=dict(samples_per_gpu=32, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            folder=f'{root_dir}/sequences',
            ann_file=f'{root_dir}/tri_trainlist.txt',
            pipeline=train_pipeline,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        folder=f'{root_dir}/sequences',
        ann_file=f'{root_dir}/tri_validlist.txt',
        pipeline=val_pipeline,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        folder=f'{root_dir}/sequences',
        ann_file=f'{root_dir}/tri_testlist.txt',
        pipeline=val_pipeline,
        test_mode=True),
)

optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
# 1604 iters == 1 epoch
total_iters = 288700
lr_config = dict(
    policy='Reduce',
    by_epoch=False,
    mode='max',
    val_metric='PSNR',
    epoch_base_valid=True,  # Support epoch base valid in iter base runner.
    factor=0.5,
    patience=5,
    cooldown=0,
    verbose=True)

checkpoint_config = dict(interval=1604, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=1604, save_image=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='TensorboardLoggerHook',
            log_dir=f'work_dirs/{exp_name}/tb_log/',
            interval=100,
            ignore_last=False,
            reset_flag=False,
            by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
