exp_name = 'tdan_vimeo90k_bix4_lr1e-4_400k'

# model settings
model = dict(
    type='TDAN',
    generator=dict(type='TDANNet'),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    lq_pixel_loss=dict(type='MSELoss', loss_weight=0.25, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=8, convert_to='y')

# dataset settings
train_dataset_type = 'SRVimeo90KDataset'
val_dataset_type = 'SRVid4Dataset'

train_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0.5, 0.5, 0.5],
        std=[1, 1, 1]),
    dict(type='PairedRandomCrop', gt_patch_size=192),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

val_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0.5, 0.5, 0.5],
        std=[1, 1, 1]),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/Vimeo-90K/BIx4',
            gt_folder='data/Vimeo-90K/GT',
            ann_file='data/Vimeo-90K/meta_info_Vimeo90K_train_GT.txt',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/Vid4/BIx4',
        gt_folder='data/Vid4/GT',
        pipeline=val_pipeline,
        ann_file='data/Vid4/meta_info_Vid4_GT.txt',
        scale=4,
        num_input_frames=5,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/SPMCS/BIx4',
        gt_folder='data/SPMCS/GT',
        pipeline=val_pipeline,
        ann_file='data/SPMCS/meta_info_SPMCS_GT.txt',
        scale=4,
        num_input_frames=5,
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, weight_decay=1e-6))

# learning policy
total_iters = 400000
lr_config = dict(policy='Step', by_epoch=False, step=[400000], gamma=0.5)

checkpoint_config = dict(interval=50000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=50000, save_image=False, gpu_collect=True)
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
