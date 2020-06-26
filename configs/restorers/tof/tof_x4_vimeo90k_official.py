# only testing the official model is supported
exp_name = 'tof_x4_vimeo90k_official'

# model settings
model = dict(
    type='EDVR',  # use the shared model with EDVR
    generator=dict(type='TOFlow', adapt_official_weights=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
val_dataset_type = 'SRVid4Dataset'

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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=3,
    drop_last=True,
    test=dict(
        type=val_dataset_type,
        lq_folder='data/Vid4/BIx4up_direct',
        gt_folder='data/Vid4/GT',
        ann_file='data/Vid4/meta_info_Vid4_GT.txt',
        num_input_frames=7,
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)

evaluation = dict(interval=5000, save_image=False, gpu_collect=False)
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
