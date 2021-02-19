model = dict(
    type='FBA',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='FBAResnetDilated', norm_cfg=dict(type='GN', num_groups=32)),
        decoder=dict(
            type='FBADecoder', norm_cfg=dict(type='GN', num_groups=32))),
    loss_alpha=dict(type='L1Loss'),
    loss_alpha_lap=dict(type='LapLoss'),
    loss_alpha_grad=dict(type='GradientLoss'),
    loss_alpha_compo=dict(type='L1CompositionLoss'),
    # loss_fb=dict(type='L1Loss', loss_weight=0.25),
    # loss_fb_compo=dict(type='FBACompLoss', loss_weight=0.25),
    # loss_fb_lap=dict(type='LapLoss', channels=3, loss_weight=0.25),
    # loss_exclusion=dict(type='ExclLoss', channels=3, loss_weight=0.25)
)
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# norm setting
norm_cfg = dict(type='GN')
num_groups = 32

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = './data/adobe_composition-1k/'
bg_dir = './data/coco/train2014'
fg_dirs = [
    data_root + 'Training_set/Adobe-licensed images/fg_extended',
    data_root + 'Training_set/Other/fg_extended'
]
alpha_dirs = [
    data_root + 'Training_set/Adobe-licensed images/alpha',
    data_root + 'Training_set/Other/alpha'
]
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='fg_extended', save_original_img=True),
    dict(
        type='RenameKeys',
        key_pairs=[('fg_extended', 'fg'), ('ori_fg_extended', 'ori_fg'),
                   ('fg_extended_path', 'fg_path'),
                   ('fg_extended_ori_shape', 'fg_ori_shape')]),
    dict(type='RandomLoadResizeBg', bg_dir=bg_dir),
    dict(type='RandomJitter'),
    dict(type='CompositeFg', fg_dirs=fg_dirs, alpha_dirs=alpha_dirs),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'fg', 'bg', 'ori_fg'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=[
        'alpha',
        'fg',
        'bg',
    ]),
    dict(
        type='Resize',
        keys=['alpha', 'fg', 'bg', 'ori_fg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='PerturbBg'),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='MergeFgAndBg'),
    dict(type='RescaleToZeroOne', keys=[
        'merged',
        'alpha',
        'fg',
        'bg',
    ]),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(type='CutEdge', mode='Train', keys=['merged', 'trimap']),
    dict(type='TransformTrimap'),
    dict(
        type='Collect',
        keys=[
            'merged', 'trimap', 'transformed_trimap', 'alpha', 'fg', 'bg',
            'ori_fg'
        ],
        meta_keys=['trimap_o']),
    dict(
        type='ImageToTensor',
        keys=[
            'merged', 'trimap', 'transformed_trimap', 'alpha', 'fg', 'bg',
            'ori_fg'
        ]),
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
    dict(
        type='LoadImageFromFile',
        key='merged',
        channel_order='rgb',
        save_original_img=True),
    dict(type='RescaleToZeroOne', keys=['merged', 'trimap']),
    dict(type='CutEdge', keys=['merged', 'trimap']),
    dict(type='TransformTrimap'),
    dict(
        type='Collect',
        keys=['merged', 'trimap', 'transformed_trimap', 'alpha'],
        meta_keys=[
            'merged_path', 'merged_ori_shape', 'ori_alpha', 'ori_trimap',
            'trimap_o', 'ori_merged'
        ]),
    dict(
        type='ImageToTensor',
        keys=['merged', 'trimap', 'transformed_trimap', 'alpha']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val_samples_per_gpu=1,
    val_workers_per_gpu=0,
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
optimizers = dict(
    type='RAdam',
    lr=1e-5,
    betas=[0.9, 0.0001],
    paramwise_cfg=dict(
        custom_keys={
            'ConvWS2d': dict(lr_mult=0.005),
            'Conv2d': dict(lr_mult=0.005),
            'GroupNorm': dict(lr_mult=1e-5)
        }))
# learning policy
lr_config = dict(policy='Step', step=[40], gamma=0.0025, by_epoch=True)

# checkpoint saving
checkpoint_config = dict(interval=2000, by_epoch=False)
evaluation = dict(interval=2000, save_image=True)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='gca'))
    ])
# yapf:enable

# runtime settings
total_iters = 200000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fba'
load_from = None
revise_keys = [(r'^', 'backbone.')]
resume_from = None
workflow = [('train', 1)]
