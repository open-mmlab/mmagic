# model settings
model = dict(
    type='DIM',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='VGG16', in_channels=4),
        decoder=dict(type='PlainDecoder')),
    pretrained='open-mmlab://mmedit/vgg16',
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
    loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5))
train_cfg = dict(train_backbone=True, train_refiner=False)
test_cfg = dict(refine=False, metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = './data/adobe_composition-1k/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged', save_original_img=True),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(
        type='RescaleToZeroOne',
        keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg', 'trimap']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'],
        meta_keys=[]),
    dict(
        type='ImageToTensor',
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        flag='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='Pad', keys=['trimap', 'merged'], mode='reflect'),
    dict(type='RescaleToZeroOne', keys=['merged', 'trimap']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'trimap'],
        meta_keys=[
            'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha', 'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'trimap']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=False,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'training_list.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizers = dict(type='Adam', lr=0.00001)
# learning policy
lr_config = dict(policy='Fixed')

# checkpoint saving
checkpoint_config = dict(interval=40000, by_epoch=False)
evaluation = dict(interval=40000, save_image=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='dim'))
    ])
# yapf:enable

# runtime settings
total_iters = 1000000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dim_stage1'
load_from = None
resume_from = None
workflow = [('train', 1)]
