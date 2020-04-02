# model settings
model = dict(
    type='DIM',
    backbone=dict(
        type='EncoderDecoder',
        encoder=dict(type='VGG16'),
        decoder=dict(type='PlainDecoder')),
    refiner=dict(type='PlainRefiner'),
    pretrained=None,  # TODO: add pretrained model
    loss_refine=dict(type='CharbonnierLoss'))
# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = './data/adobe_composition-1k/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
# TODO: add data preparation and update data pipeline
train_pipeline = [
    dict(type='LoadAlpha'),
    dict(type='LoadFgAndBg'),
    dict(type='LoadMerged'),
    dict(type='CropAroundSemiTransparent', crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(
        type='GenerateTrimap',
        kernel_size=(2, 5),
        iterations=(5, 15),
        symmetric=True),
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
    dict(type='LoadAlpha'),
    dict(type='LoadTrimap'),
    dict(type='LoadMerged'),
    dict(type='Reshape', keys=['alpha', 'trimap', 'merged']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'alpha', 'trimap'],
        meta_keys=[
            'img_name', 'test_trans', 'ori_shape', 'ori_alpha', 'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap']),
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
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
optimizer = dict(type='Adam', lr=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmmat')),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 25
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dim_stage2'
load_from = './work_dirs/dim_stage1/latest.pth'
resume_from = None
workflow = [('train', 1)]
train_cfg = dict(train_backbone=False, train_refiner=True)
test_cfg = dict(refine=True, metrics=['SAD', 'MSE'])
evaluation = dict()
