custom_imports = dict(
    imports=['mmedit.core.optimizer.radam'], allow_failed_imports=False)

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
    loss_fb=dict(type='L1Loss', loss_weight=0.25),
    loss_fb_compo=dict(type='FBACompLoss', loss_weight=0.25),
    loss_fb_lap=dict(type='LapLoss', channels=3, loss_weight=0.25),
    loss_exclusion=dict(type='ExclLoss', channels=3, loss_weight=0.25))
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# norm setting
norm_cfg = dict(type='GN')
num_groups = 32

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = './data/adobe_composition-1k/'
bg_dir = './data/coco/'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [  # Training data processing pipeline.
    dict(
        type='LoadImageFromFile',  # Load alpha matte from file.
        key='alpha',
        flag='grayscale'
    ),  # Load as grayscale image which has shape (height, width).
    dict(
        type='LoadImageFromFile',  # Load image from file.
        key='fg'
    ),  # Key of image to load. The pipeline will read fg from path `fg_path`.
    dict(
        type='LoadImageFromFile',  # Load image from file.
        key='bg'
    ),  # Key of image to load. The pipeline will read bg from path `bg_path`.
    dict(
        type='LoadImageFromFile',  # Load image from file.
        key='merged'),
    dict(type='CompositeFg', fg_dirs='', alpha_dirs=''),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'fg', 'bg'],  # Images to crop.
        crop_sizes=[320, 480, 640]),  # Candidate crop size.
    dict(
        type='Flip',  # Augmentation pipeline that flips the images.
        keys=['alpha', 'merged', 'fg', 'bg']),  # Images to be flipped.
    dict(
        type='Resize',  # Augmentation pipeline that resizes the images.
        keys=['alpha', 'merged', 'fg', 'bg'],  # Images to be resized.
        scale=(320, 320),  # Target size.
        keep_ratio=False
    ),  # Whether to keep the ratio between height and width.
    dict(type='PerturbBg'),
    # dict(type='RandomJitter'),
    dict(
        type='GenerateTrimap',  # Generate trimap from alpha matte.
        kernel_size=(1, 30)),  # Kernel size range of the erode/dilate kernel.
    dict(
        type='RescaleToZeroOne',  # Rescale images from [0, 255] to [0, 1].
        keys=['merged', 'alpha', 'fg', 'bg']),  # Images to be rescaled.
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(type='CutEdge', mode='Train', keys=['merged', 'trimap']),
    dict(type='TransformTrimap'),
    dict(
        type='Collect',
        keys=['merged', 'trimap', 'transformed_trimap', 'alpha', 'fg', 'bg'],
        meta_keys=['merged_path', 'merged_ori_shape', 'trimap_o']),
    dict(
        type='ImageToTensor',  # Convert images to tensor.
        keys=['merged', 'trimap', 'transformed_trimap', 'alpha', 'fg',
              'bg']),  # Images to be converted to Tensor.
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
        type='LoadImageFromFile',  # Load image from file.
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
optimizers = dict(type='RAdam', lr=1e-5, betas=[0.9, 0.0001])
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
resume_from = '/nfs/Code/mmediting/work_dirs/fba/iter_2000.pth'
workflow = [('train', 1)]
