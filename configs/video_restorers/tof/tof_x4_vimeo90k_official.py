# only testing the official model is supported
from ..dataset.val_vid4_bix4_up import val_dataloader

_base_ = [
    '../../default_runtime.py',
]

experiment_name = 'tof_x4_vimeo90k_official'

# model settings
model = dict(
    type='EDVR',  # use the shared model with EDVR
    generator=dict(type='TOFlow', adapt_official_weights=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        input_view=(1, -1, 1, 1),
        output_view=(-1, 1, 1)))

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', imdecode_backend='unchanged'),
    dict(type='ToTensor', keys=['img']),
    dict(type='PackEditInputs')
]

test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        out_dir='s3://ysli/tof/',
        by_epoch=False))
