# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '../_base_/default_runtime.py'
]

model = dict(
    type='DeblurGanV2',
    generator=dict(
        type='FPNMobileNet',
        norm_layer='instance',
        output_ch=3,
        num_filters=64,
        num_filters_fpn=128,
        pretrained="mobilenetv2.pth.tar",
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
)

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(type='PackEditInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(type='PackEditInputs')
]

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader