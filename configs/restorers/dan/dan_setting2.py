exp_name = 'dan_setting2_lr6e25-6_320'

scale = 4
# model settings
model = dict(
    type='DAN',
    generator=dict(
        type='DAN',
        nf=64,
        nb=40,
        input_para=10,
        loop=4,
        kernel_size=21,
        pca_matrix_path='/home/ivdai/Temp/xxx/mmediting/tools/data/super-resolution/div2k/pca_matrix/pca_aniso_matrix_x4.pth'),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(pca_matrix_path='/home/ivdai/Temp/xxx/mmediting/tools/data/super-resolution/div2k/pca_matrix/pca_aniso_matrix_x4.pth',
                 scale=scale,
                 degradation=dict(
                    random_kernel = True,
                    ksize=21,
                    code_length=10,
                    sig_min=0.6,
                    sig_max=5.0,
                    rate_iso=0,
                    random_disturb=True
                 ))
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRFolderGTDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['gt']),
    dict(
        type='Normalize',
        keys=['gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Crop', keys=['gt'], crop_size=(256, 256)),
    dict(
        type='Flip', keys=['gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['gt'], meta_keys=['gt_path']),
    dict(type='ImageToTensor', keys=['gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=8, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            gt_folder='/home/ivdai/Temp/xxx/dataset/DF2K_train_HR_sub',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='/home/ivdai/Temp/xxx/dataset/DIV2KRK/lr_x4',
        gt_folder='/home/ivdai/Temp/xxx/dataset/DIV2KRK/gt',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='/home/ivdai/Temp/xxx/dataset/DIV2KRK/lr_x4',
        gt_folder='/home/ivdai/Temp/xxx/dataset/DIV2KRK/gt',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=6.25e-6, betas=(0.9, 0.999)))

# learning policy
total_iters = 400000
lr_config = dict(policy='Step', by_epoch=False, step=[100000,200000,300000], gamma=0.5)

checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=1000, save_image=False, gpu_collect=True)
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
load_from = '/home/ivdai/Temp/xxx/mmediting/danx4_l1_256/iter_40000.pth'
resume_from = None
workflow = [('train', 1)]
