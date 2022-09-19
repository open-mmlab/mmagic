model = dict(
    type='FusionModel',
    stage='full',
    ngf=64,
    output_nc=2,
    avg_loss_alpha=.986,
    ab_norm=110.,
    l_norm=100.,
    l_cent=50.,
    sample_Ps=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    mask_cent=.5,
    init_type='normal',
    fusion_weight_path='checkpoints/coco_finetuned_mask_256_ffs',
    which_direction='AtoB',
    loss=dict(type='HuberLoss', delta=.01),
    instance_model=dict(
        type='SIGGRAPHGenerator',
        input_nc=4,
        output_nc=2,
    ))

train_cfg = dict(disc_step=1)
test_cfg = dict(metrics=['psnr', 'ssim'])
input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'),
    dict(type='GenGrayColorPil', stage='full', keys=['rgb_img', 'gray_img']),
    dict(
        type='Resize',
        keys=['rgb_img', 'gray_img'],
        scale=input_shape,
        keep_ratio=False,
        interpolation='nearest'),
    dict(
        type='Collect',
        keys=['rgb_img', 'gray_img'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['rgb_img', 'gray_img'])
]

dataset_type = 'COCOStuff_Full_Dataset'
data_root = '/mnt/j/DataSet/cocostuff/train2017'
ann_file_path = '/mnt/j/DataSet/cocostuff/'

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=f'{ann_file_path}/img_list.txt',
        data_prefix=data_root,
        pipeline=train_pipeline,
        test_mode=False))

optimizers = dict(generator=dict(type='Adam', lr=0.0001, betas=(0.9, 0.999)), )
lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=10000)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])

visual_config = dict(
    type='VisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=[
        'gray', 'real', 'fake_reg', 'hint', 'real_ab', 'fake_ab_reg'
    ],
)

total_iters = 500002
dist_params = dict(backend='nccl')
load_from = None
resume_from = None
work_dir = '..'
log_level = 'INFO'
workflow = [('train', 10000)]
exp_name = 'Instance-aware_full'
find_unused_parameters = True
