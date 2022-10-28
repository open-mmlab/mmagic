model = dict(
    type='PConvInpaintor',
    encdec=dict(
        type='PConvEncoderDecoder',
        encoder=dict(
            type='PConvEncoder',
            norm_cfg=dict(type='SyncBN', requires_grad=False),
            norm_eval=True),
        decoder=dict(type='PConvDecoder', norm_cfg=dict(type='SyncBN'))),
    disc=None,
    loss_composed_percep=dict(
        type='PerceptualLoss',
        vgg_type='vgg16',
        layer_weights={
            '4': 1.,
            '9': 1.,
            '16': 1.,
        },
        perceptual_weight=0.05,
        style_weight=120,
        pretrained=('torchvision://vgg16')),
    loss_out_percep=True,
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=6.,
    ),
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.,
    ),
    loss_tv=dict(
        type='MaskedTVLoss',
        loss_weight=0.1,
    ),
    pretrained=None)

train_cfg = dict(disc_step=0)
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

dataset_type = 'ImgInpaintingDataset'
input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'),
    dict(
        type='LoadMask',
        mask_mode='irregular',
        mask_config=dict(
            num_vertices=(4, 10),
            max_angle=6.0,
            length_range=(20, 128),
            brush_width=(10, 45),
            area_ratio_range=(0.15, 0.65),
            img_shape=input_shape)),
    dict(
        type='Crop',
        keys=['gt_img'],
        crop_size=(384, 384),
        random_crop=True,
    ),
    dict(
        type='Resize',
        keys=['gt_img'],
        scale=input_shape,
        keep_ratio=False,
    ),
    dict(
        type='Normalize',
        keys=['gt_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='GetMaskedImage'),
    dict(
        type='Collect',
        keys=['gt_img', 'masked_img', 'mask'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask'])
]

test_pipeline = train_pipeline
data_root = 'data/places365'

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=12, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/train_places_img_list_total.txt',
        data_prefix=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/val_places_img_list.txt',
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=(f'{data_root}/val_places_img_list.txt'),
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True))

optimizers = dict(generator=dict(type='Adam',
                                 lr=0.00005))  # second stage training

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=50000)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])

visual_config = dict(
    type='MMEditVisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=['gt_img', 'masked_img', 'fake_res', 'fake_img'],
)

evaluation = dict(interval=50000)

total_iters = 500000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pconv'
load_from = None
resume_from = None
workflow = [('train', 10000)]
exp_name = 'pconv_256x256_stage2_4x2_places'
find_unused_parameters = False
