_base_ = [
    '../_base_/models/base_pconv.py', '../_base_/inpaint_default_runtime.py',
    '../_base_/datasets/celeba.py'
]

experiment_name = 'pconv_stage2_4xb2_celeba-256x256'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    train_cfg=dict(
        disc_step=0,
        start_iter=0,
    ),
    encdec=dict(
        type='PConvEncoderDecoder',
        encoder=dict(
            type='PConvEncoder',
            norm_cfg=dict(type='SyncBN', requires_grad=False),
            norm_eval=True),
        decoder=dict(type='PConvDecoder', norm_cfg=dict(type='SyncBN'))),
)

input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(
        type='LoadMask',
        mask_mode='irregular',
        mask_config=dict(
            num_vertices=(4, 10),
            max_angle=6.0,
            length_range=(20, 128),
            brush_width=(10, 45),
            area_ratio_range=(0.15, 0.65),
            img_shape=input_shape)),
    dict(
        type='Crop',
        keys=['gt'],
        crop_size=(384, 384),
        random_crop=True,
    ),
    dict(
        type='Resize',
        keys=['gt'],
        scale=input_shape,
        keep_ratio=False,
    ),
    dict(type='GetMaskedImage'),
    dict(type='PackInputs'),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=2,
    sampler=dict(shuffle=False),
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=300002,
    val_interval=50000,
)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.00005))

checkpoint = dict(
    type='CheckpointHook', interval=50000, by_epoch=False, out_dir=save_dir)

# load_from = 'pconv_stage1_8xb1_celeba-256x256/iter_800002.pth'
