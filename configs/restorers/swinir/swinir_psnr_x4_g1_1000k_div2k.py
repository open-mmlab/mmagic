exp_name = 'swinir_psnr_x4_g1_1000k_div2k'

scale = 4
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='SwinIR',
        in_channels=3,
        img_size=48,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dims=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
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
    dict(type='PairedRandomCrop', gt_patch_size=96),
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
    workers_per_gpu=128,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X4_sub',
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',
            #ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
            ann_file='data/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb/meta_info.txt',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        #lq_folder='data/val_set5/Set5_bicLRx4',
        #gt_folder='data/val_set5/Set5',
        lq_folder='/nas/k8s/dev/mlops/dataset-artifacts/super_resolution/version_1/data/set5/bicLRx4',
        gt_folder='/nas/k8s/dev/mlops/dataset-artifacts/super_resolution/version_1/data/set5/HR',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/Set14/LRbicx4',
        gt_folder='data/Set14/original',
        #lq_folder='/nas/k8s/dev/mlops/dataset-artifacts/super_resolution/version_1/data/set14/bicLRx4',
        #gt_folder='/nas/k8s/dev/mlops/dataset-artifacts/super_resolution/version_1/data/set14/HR',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
# keep training
total_iters = 1000000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[250000, 400000, 450000, 4750000, 500000],
    gamma=0.5)

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

# fp16 settings
fp16 = dict(loss_scale=512.0)
