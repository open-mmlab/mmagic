# model settings
model = dict(
    type='GCA',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='ResGCAEncoder',
            block='BasicBlock',
            layers=[3, 4, 4, 2],
            in_channels=4,
            with_spectral_norm=True),
        decoder=dict(
            type='ResGCADecoder',
            block='BasicBlockDec',
            layers=[2, 3, 3, 2],
            with_spectral_norm=True)),
    loss_alpha=dict(type='L1Loss'),
    pretrained='open-mmlab://mmedit/res34_en_nomixup')
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = 'data/adobe_composition-1k'
bg_dir = './data/coco/train2017'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='merged'),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged']),
    dict(
        type='Resize',
        keys=['alpha', 'merged'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='RescaleToZeroOne', keys=['merged', 'alpha']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(type='Collect', keys=['merged', 'alpha', 'trimap'], meta_keys=[]),
    dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap']),
    dict(type='FormatTrimap', to_onehot=False),
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
    dict(type='RescaleToZeroOne', keys=['merged']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'trimap'],
        meta_keys=[
            'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha', 'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'trimap']),
    dict(type='FormatTrimap', to_onehot=False),
]
data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=10, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/training_list.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizers = dict(type='Adam', lr=4e-4, betas=[0.5, 0.999])
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001)

# checkpoint saving
checkpoint_config = dict(interval=2000, by_epoch=False)
evaluation = dict(interval=2000, save_image=False, gpu_collect=False)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='gca'))
    ])

# runtime settings
total_iters = 200000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/gca'
load_from = None
resume_from = None
workflow = [('train', 1)]
