_base_ = [
    '../_base_/datasets/comp1k.py', '../_base_/matting_default_runtime.py'
]

save_dir = './work_dirs/'
experiment_name = 'dim_stage1-v16_1xb1-1000k_comp1k'

# model settings
model = dict(
    type='DIM',
    data_preprocessor=dict(
        type='MattorPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        proc_trimap='rescale_to_zero_one',
    ),
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='VGG16',
            in_channels=4,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://mmedit/vgg16')),
        decoder=dict(type='PlainDecoder')),
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
    loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5),
    train_cfg=dict(train_backbone=True, train_refiner=False),
    test_cfg=dict(
        refine=False,
        resize_method='pad',
        resize_mode='reflect',
        size_divisor=32,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged'),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        color_type='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='PackInputs'),
]

train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1_000_000,
    val_interval=40000,
)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.00001),
)

# checkpoint saving
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=40000, out_dir=save_dir))
