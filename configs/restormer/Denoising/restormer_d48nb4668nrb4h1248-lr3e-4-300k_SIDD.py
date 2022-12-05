_base_ = '../../_base_/default_runtime.py'

experiment_name = 'restormer_d48nb4668nrb4h1248-lr3e-4-300k_SIDD'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 1
# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='Restormer',
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR'], crop_border=scale),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackEditInputs')
]
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=8,
    batch_size=8,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='Test100', task_name='deraining'),
        data_root='/home/ywp/zwb/data/Test/Deraining/Test2800',
        data_prefix=dict(img='input', gt='target'),
        ann_file='meta_info_test2800_test.txt',
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='Test100', task_name='deraining'),
        data_root='/home/ywp/zwb/data/Test/Deraining/Test2800',
        ann_file='meta_info_test2800_test.txt',
        data_prefix=dict(img='input', gt='target'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]

test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[92000, 208000],
    restart_weights=[1, 1],
    eta_min=[0.0003, 0.000001])

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['pred_img'],
    bgr2rgb=False)

randomness = dict(seed=10, diff_rank_seed=True)
