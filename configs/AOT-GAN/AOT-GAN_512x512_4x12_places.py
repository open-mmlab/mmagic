_base_ = [
    '../__base__/models/AOT-GAN_base.py',
    '../__base__/inpaint_default_runtime.py', '../__base__/datasets/places.py'
]

model = dict(train_cfg=dict(
    disc_step=1,
    start_iter=0,
))

input_shape = (512, 512)

mask_root = 'data/pconv_mask'
train_pipeline = [
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(
        type='LoadMask',
        mask_mode='set',
        mask_config=dict(
            mask_list_file=f'{mask_root}/train_mask_list.txt',
            prefix=mask_root,
            io_backend='disk',
            flag='unchanged',
            color_type='unchanged',
            file_client_kwargs=dict())),
    dict(
        type='RandomResizedCrop',
        keys=['gt'],
        crop_size=input_shape,
    ),
    dict(type='Flip', keys=['gt', 'mask'], direction='horizontal'),
    dict(
        type='Resize',
        keys=['mask'],
        scale=input_shape,
        keep_ratio=False,
        interpolation='nearest'),
    dict(type='RandomRotation', keys=['mask'], degrees=(0.0, 45.0)),
    dict(
        type='ColorJitter',
        keys=['gt'],
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='GetMaskedImage'),
    dict(type='PackEditInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(
        type='LoadMask',
        mask_mode='set',
        mask_config=dict(
            mask_list_file=f'{mask_root}/mask_0.5-0.6_list.txt',
            prefix=mask_root,
            io_backend='disk',
            color_type='unchanged',
            flag='unchanged',
            file_client_kwargs=dict())),
    dict(
        type='RandomResizedCrop',
        keys=['gt'],
        crop_size=(512, 512),
    ),
    dict(type='GetMaskedImage'),
    dict(type='PackEditInputs'),
]

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
