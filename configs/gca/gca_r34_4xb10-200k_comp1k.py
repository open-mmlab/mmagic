_base_ = [
    '../_base_/datasets/comp1k.py', '../_base_/matting_default_runtime.py'
]

experiment_name = 'gca_r34_4xb10-200k_comp1k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='GCA',
    data_preprocessor=dict(
        type='MattorPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        proc_inputs='normalize',
        proc_trimap='as_is',
        proc_gt='rescale_to_zero_one',
    ),
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='ResGCAEncoder',
            block='BasicBlock',
            layers=[3, 4, 4, 2],
            in_channels=6,
            with_spectral_norm=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://mmedit/res34_en_nomixup')),
        decoder=dict(
            type='ResGCADecoder',
            block='BasicBlockDec',
            layers=[2, 3, 3, 2],
            with_spectral_norm=True)),
    loss_alpha=dict(type='L1Loss'),
    test_cfg=dict(
        resize_method='pad',
        resize_mode='reflect',
        size_divisor=32,
    ))

# dataset settings
data_root = 'data/adobe_composition-1k'
bg_dir = 'data/coco/train2017'
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='RandomLoadResizeBg', bg_dir=bg_dir),
    dict(
        type='CompositeFg',
        fg_dirs=[
            f'{data_root}/Training_set/Adobe-licensed images/fg',
            f'{data_root}/Training_set/Other/fg'
        ],
        alpha_dirs=[
            f'{data_root}/Training_set/Adobe-licensed images/alpha',
            f'{data_root}/Training_set/Other/alpha'
        ]),
    dict(
        type='RandomAffine',
        keys=['alpha', 'fg'],
        degrees=30,
        scale=(0.8, 1.25),
        shear=10,
        flip_ratio=0.5),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='CropAroundCenter', crop_size=512),
    dict(type='RandomJitter'),
    dict(type='MergeFgAndBg'),
    dict(type='FormatTrimap', to_onehot=True),
    dict(type='PackEditInputs'),
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
    dict(type='FormatTrimap', to_onehot=True),
    dict(type='PackEditInputs'),
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=200_000,
    val_interval=10_000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=4e-4, betas=[0.5, 0.999]))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=5000,
        by_epoch=False,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=200_000,  # TODO, need more check
        eta_min=0,
        begin=0,
        end=200_000,
        by_epoch=False,
    )
]

# checkpoint saving
# inheritate from _base_

# runtime settings
# inheritate from _base_
