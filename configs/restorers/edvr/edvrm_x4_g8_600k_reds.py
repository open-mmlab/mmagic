exp_name = 'edvrm_x4_g8_600k_reds'

# model settings
model = dict(
    type='EDVR',
    generator=dict(
        type='EDVRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_frames=5,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=10,
        center_frame_idx=2,
        with_tsa=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'))
# model training and testing settings
train_cfg = dict(tsa_iter=50000)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'SRREDSDataset'
val_dataset_type = 'SRREDSDataset'
train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
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
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
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
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/REDS/train_sharp_bicubic/X4',
            gt_folder='data/REDS/train_sharp',
            ann_file='data/REDS/meta_info_REDS_GT.txt',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=4,
            val_partition='REDS4',
            test_mode=False)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/REDS/train_sharp_bicubic/X4',
        gt_folder='data/REDS/train_sharp',
        ann_file='data/REDS/meta_info_REDS_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/REDS/train_sharp_bicubic/X4',
        gt_folder='data/REDS/train_sharp',
        ann_file='data/REDS/meta_info_REDS_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 600000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[50000, 100000, 150000, 150000, 150000],
    restart_weights=[1, 1, 1, 1, 1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=50000, save_image=False, gpu_collect=True)
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
load_from = 'work_dirs/201_EDVRM_woTSA/iter_600000.pth'
resume_from = None
workflow = [('train', 1)]
