exp_name = 'cain_b5_320k_vimeo-triplet'

# model settings
model = dict(
    type='CAIN',
    generator=dict(type='CAINNet'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0, convert_to='y')

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
    dict(type='RescaleToZeroOne', keys=['inputs', 'target']),
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
    dict(type='TemporalReverse', keys=['inputs'], reverse_ratio=0.5),
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

root_dir = 'data/vimeo_triple'
data = dict(
    workers_per_gpu=4,
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

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 320000
lr_config = dict(
    policy='Step', by_epoch=False, step=[80000, 160000, 240000], gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=500, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
