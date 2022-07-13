from ..dataset.val_reds_reds4 import val_dataloader

_base_ = [
    '../../default_runtime.py',
    '../dataset/train_reds_reds4.py',
    '../dataset/val_reds_reds.py',
]

scale = 4

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', color_type='unchanged'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackEditInputs')
]

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=600_000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)),
    ))

# learning policy
# lr_config = dict(
#     policy='CosineRestart',
#     by_epoch=False,
#     periods=[50000, 100000, 150000, 150000, 150000],
#     restart_weights=[1, 1, 1, 1, 1],
#     min_lr=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        out_dir='s3://ysli/edvr/',
        by_epoch=False))
