exp_name = 'dic_x8c48b6_g4_150k_CelebAHQ'
scale = 8

# model settings
model = dict(
    type='DIC',
    generator=dict(
        type='DICNet', in_channels=3, out_channels=3, mid_channels=48),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRFacialLandmarkDataset'
val_dataset_type = 'SRFolderGTDataset'
test_dataset_type = 'SRFolderGTDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb',
        backend='cv2'),
    dict(
        type='Resize',
        scale=(128, 128),
        keys=['gt'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=1 / 8,
        keep_ratio=True,
        keys=['gt'],
        output_keys=['lq'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='GenerateHeatmap',
        keypoint='landmark',
        ori_size=256,
        target_size=32,
        sigma=1.),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[129.795, 108.12, 96.39],
        std=[255, 255, 255]),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'heatmap']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'heatmap', 'landmark'],
        meta_keys=['gt_path'])
]
valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb',
        backend='cv2'),
    dict(
        type='Resize',
        scale=(128, 128),
        keys=['gt'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Resize',
        scale=1 / 8,
        keep_ratio=True,
        keys=['gt'],
        output_keys=['lq'],
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[129.795, 108.12, 96.39],
        std=[255, 255, 255]),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['gt_path'])
]
test_pipeline = valid_pipeline

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=60,
        dataset=dict(
            type=train_dataset_type,
            gt_folder='data/celeba-hq/train/',
            ann_file='data/celeba-hq/train_info_list_256.npy',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        gt_folder='data/celeba-hq/val/',
        pipeline=valid_pipeline,
        scale=scale),
    test=dict(
        type=test_dataset_type,
        gt_folder='data/celeba-hq/val/',
        pipeline=valid_pipeline,
        scale=scale))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1.e-4))

# learning policy
total_iters = 150000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[10000, 20000, 40000, 80000],
    gamma=0.5)

checkpoint_config = dict(interval=2000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=2000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
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
