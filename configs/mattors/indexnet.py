# model settings
model = dict(
    type='IndexNet',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='IndexNetEncoder'),
        decoder=dict(type='IndexNetDecoder')),
    loss_alpha=dict(type='L1Loss'),
    loss_comp=dict(type='L1CompositionLoss'))
# model training and testing settings
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE'])

# data settings
dataset_type = 'AdobeComp1kDataset'
data_root = './data/adobe_composition-1k/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
# TODO: add data preparation and update data pipeline
train_pipeline = None
test_pipeline = [
    dict(
        type='LoadAlpha',
        key='alpha',
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        flag='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='RescaleToZeroOne', keys=['merged', 'alpha']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Resize',
        keys=['alpha', 'trimap'],
        size_factor=32,
        interpolation='nearest'),
    dict(
        type='Resize',
        keys=['merged'],
        size_factor=32,
        interpolation='bicubic'),
    dict(
        type='Collect',
        keys=['merged', 'alpha', 'trimap'],
        meta_keys=[
            'merged_path', 'interpolation', 'ori_shape', 'ori_alpha',
            'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap']),
]
data = dict(
    # train
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'training_list.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    # validation
    val_samples_per_gpu=1,
    val_workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    # test
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizers = None
# learning policy
lr_config = None

# checkpoint saving
checkpoint_config = dict(interval=40000, by_epoch=False)
evaluation = dict(interval=40000, save_image=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='indexnet')),
    ])
# yapf:enable

# runtime settings
total_iters = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/indexnet'
load_from = None
resume_from = None
workflow = [('train', 1)]
