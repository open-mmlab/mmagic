_base_ = [
    '../../default_runtime.py',
    '../dataset/train_vimeo_seq_bix4.py',
    '../dataset/val_vid4_bix4.py',
    '../dataset/val_spmcs_bix4.py',
]

exp_name = 'tdan_vimeo90k_bix4_lr1e-4_400k'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='TDAN',
    generator=dict(
        type='TDANNet',
        in_channels=3,
        mid_channels=64,
        out_channels=3,
        num_blocks_before_align=5,
        num_blocks_after_align=10),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    lq_pixel_loss=dict(type='MSELoss', loss_weight=0.25, reduction='mean'),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        std=[255, 255, 255],
        input_view=(1, -1, 1, 1),
        output_view=(-1, 1, 1)))
# test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=8, convert_to='y')

val_evaluator = [
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackEditInputs')
]

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-6),
    ))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=400_000, val_interval=50000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# No learning policy

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=50000,
        save_optimizer=True,
        out_dir='s3://ysli/edvr/',
        by_epoch=False))
