exp_name = 'flavr_in4out1_g8b4_vimeo90k_septuplet'

# model settings
model = dict(
    type='FLAVR',
    generator=dict(
        type='FLAVRNet',
        num_input_frames=4,
        num_output_frames=1,
        mid_channels_list=[512, 256, 128, 64],
        encoder_layers_list=[2, 2, 2, 2],
        bias=False,
        norm_cfg=None,
        join_type='concat',
        up_mode='transpose'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM', 'MAE'], crop_border=0)

# dataset settings
train_dataset_type = 'VFIVimeo90K7FramesDataset'
val_dataset_type = 'VFIVimeo90K7FramesDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='inputs',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='LoadImageFromFileList',
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
    dict(type='FramesToTensor', keys=['inputs', 'target']),
    dict(
        type='Collect',
        keys=['inputs', 'target'],
        meta_keys=['inputs_path', 'target_path', 'key'])
]

valid_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='inputs',
        channel_order='rgb',
        backend='pillow'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='target',
        channel_order='rgb',
        backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['inputs', 'target']),
    dict(type='FramesToTensor', keys=['inputs', 'target']),
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

root_dir = 'data/vimeo90k'
data = dict(
    workers_per_gpu=16,
    train_dataloader=dict(samples_per_gpu=4),  # 8 gpu
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),

    # train
    train=dict(
        type=train_dataset_type,
        folder=f'{root_dir}/GT',
        ann_file=f'{root_dir}/sep_trainlist.txt',
        pipeline=train_pipeline,
        input_frames=[1, 3, 5, 7],
        target_frames=[4],
        test_mode=False),
    # val
    val=dict(
        type=train_dataset_type,
        folder=f'{root_dir}/GT',
        ann_file=f'{root_dir}/sep_testlist.txt',
        pipeline=valid_pipeline,
        input_frames=[1, 3, 5, 7],
        target_frames=[4],
        test_mode=True),
    # test
    test=dict(
        type=train_dataset_type,
        folder=f'{root_dir}/GT',
        ann_file=f'{root_dir}/sep_testlist.txt',
        pipeline=valid_pipeline,
        input_frames=[1, 3, 5, 7],
        target_frames=[4],
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 1000000  # >=200*64612/64
lr_config = dict(
    policy='Reduce',
    by_epoch=False,
    mode='max',
    val_metric='PSNR',
    epoch_base_valid=True,  # Support epoch base valid in iter base runner.
    factor=0.5,
    patience=10,
    cooldown=20,
    verbose=True)

checkpoint_config = dict(interval=2020, save_optimizer=True, by_epoch=False)

evaluation = dict(interval=2020, save_image=False, gpu_collect=True)
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
