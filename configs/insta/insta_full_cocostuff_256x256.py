_base_ = [
    '../_base_/default_runtime.py'
]

exp_name = 'Instance-aware_full'
save_dir = './'
work_dir = '..'

model = dict(
    type='INSTA',
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    instance_model=dict(
        type='SIGGRAPHGenerator',
        input_nc=4,
        output_nc=2,
        norm_type='batch'
    ),
    insta_stage='full',
    ngf=64,
    output_nc=2,
    avg_loss_alpha=.986,
    ab_norm=110.,
    ab_max=110.,
    ab_quant=10.,
    l_norm=100.,
    l_cent=50.,
    sample_Ps=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    mask_cent=.5,
    which_direction='AtoB',
    loss=dict(type='HuberLoss', delta=.01),
)

input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='GenGrayColorPil', stage='full', keys=['rgb_img', 'gray_img']),
    dict(
        type='Resize',
        keys=['rgb_img', 'gray_img'],
        scale=input_shape,
        keep_ratio=False,
        interpolation='nearest'),
    dict(type='RescaleToZeroOne', keys=['rgb_img', 'gray_img']),
    dict(type='PackEditInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='GenMaskRCNNBbox', stage='test_fusion', finesize=256),
    dict(type='Resize',
         keys=['img'],
         scale=(256, 256),
         keep_ratio=False
         ),
    dict(type='PackEditInputs'),
]

dataset_type = 'CocoDataset'
data_root = '/mnt/j/DataSet/cocostuff'
ann_file_path = '/mnt/j/DataSet/cocostuff'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + '/train2017',
        data_prefix=dict(gt='data_large'),
        ann_file=f'{ann_file_path}/img_list.txt',
        pipeline=train_pipeline,
        test_mode=False))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + '/train2017',
        data_prefix=dict(gt='data_large'),
        ann_file=f'{ann_file_path}/train_annotation.json',
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = [dict(type='PSNR'), dict(type='SSIM')]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=500002,
    val_interval=50000,
)

val_dataloader = test_dataloader
val_evaluator = test_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9))),
    disc=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9))))

param_scheduler = dict(
    # todo engine中暂时还没有这个
    type='LambdaLR',
    by_epoch=False,
)

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=[
        'gray', 'real', 'fake_reg', 'hint', 'real_ab', 'fake_ab_reg'
    ],
    bgr2rgb=False)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
