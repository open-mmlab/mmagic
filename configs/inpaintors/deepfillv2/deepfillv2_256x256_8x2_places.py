_base_ = [
    'deepfillv2_base.py', '../default_runtime.py', '../datasets/places.py'
]

model = dict(train_cfg=dict(disc_step=1, start_iter=0))

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
    dict(type='PackEditInputs'),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=16,
    sampler=dict(shuffle=False),
    dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1000003,
    val_interval=50000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint = dict(type='CheckpointHook', interval=50000, by_epoch=False)
