# data path
data_root = './data/adobe_composition-1k/'
bg_dir = './data/coco/train2017'
# model settings
model = dict(
    type='GCA',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='ResGCAEncoder',
            block='BasicBlock',
            layers=[3, 4, 4, 2],
            in_channels=6,
            with_spectral_norm=True),
        decoder=dict(
            type='ResGCADecoder',
            block='BasicBlockDec',
            layers=[2, 3, 3, 2],
            with_spectral_norm=True)),
    loss_alpha=dict(type='L1Loss'),
    pretrained='./weights/model_best_resnet34_En_nomixup_mmedit.pth')
# dataset settings
dataset_type = 'AdobeComp1kDataset'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadAlpha', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='RandomLoadResizeBg', bg_dir=bg_dir),
    dict(
        type='CompositeFg',
        fg_dir='./data/adobe_composition-1k_bk/' + 'Training_set/all/fg',
        alpha_dir='./data/adobe_composition-1k_bk/' + 'Training_set/all/alpha',
        fg_ext='jpg',
        alpha_ext='jpg'),
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
    dict(type='RescaleToZeroOne', keys=['merged', 'alpha']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(type='Collect', keys=['merged', 'alpha', 'trimap'], meta_keys=[]),
    dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap']),
    dict(type='FormatTrimap', to_onehot=True),
]
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
    dict(type='Pad', keys=['alpha', 'trimap', 'merged'], mode='reflect'),
    dict(type='RescaleToZeroOne', keys=['merged', 'alpha', 'ori_alpha']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'trimap'],
        meta_keys=[
            'merged_path', 'pad', 'ori_shape', 'ori_alpha', 'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'trimap']),
    dict(type='FormatTrimap', to_onehot=True),
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    val_samples_per_gpu=1,
    val_workers_per_gpu=4,
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
optimizers = dict(type='Adam', lr=4e-4, betas=[0.5, 0.999])
# learning policy
lr_config = dict(
    policy='CosineAnealing',
    min_lr=0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
)
# checkpoint saving
checkpoint_config = dict(interval=2000, by_epoch=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='IterTextLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='gca')),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_iters = 200000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/gca'
load_from = None
resume_from = None
workflow = [('train', 1)]
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE'])
evaluation = dict(interval=2000, save_image=False, gpu_collect=True)
