_base_ = [
    '../_base_/models/base_gl.py', '../_base_/inpaint_default_runtime.py',
    '../_base_/datasets/places.py'
]

experiment_name = 'gl_8xb12_places-256x256'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    train_cfg=dict(
        disc_step=1,
        iter_tc=90000,
        iter_td=100000,
        start_iter=350000,
        local_size=(128, 128)), )

input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(
        type='LoadMask',
        mask_mode='bbox',
        mask_config=dict(
            max_bbox_shape=(128, 128),
            max_bbox_delta=40,
            min_margin=20,
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
    batch_size=12,
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
    max_iters=500002,
    val_interval=50000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# runtime settings
# inheritate from _base_
