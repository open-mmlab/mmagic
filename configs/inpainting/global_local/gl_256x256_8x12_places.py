model = dict(
    type='GLInpaintor',
    encdec=dict(
        type='GLEncoderDecoder',
        encoder=dict(type='GLEncoder', norm_cfg=dict(type='SyncBN')),
        decoder=dict(type='GLDecoder', norm_cfg=dict(type='SyncBN')),
        dilation_neck=dict(
            type='GLDilationNeck', norm_cfg=dict(type='SyncBN'))),
    disc=dict(
        type='GLDiscs',
        global_disc_cfg=dict(
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 4 * 4,
            fc_out_channels=1024,
            num_convs=6,
            norm_cfg=dict(type='SyncBN'),
        ),
        local_disc_cfg=dict(
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 4 * 4,
            fc_out_channels=1024,
            num_convs=5,
            norm_cfg=dict(type='SyncBN'),
        ),
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.001,
    ),
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    pretrained=None)

train_cfg = dict(
    disc_step=1,
    iter_tc=90000,
    iter_td=100000,
    start_iter=350000,
    local_size=(128, 128))
test_cfg = dict(metrics=['l1'])

dataset_type = 'ImgInpaintingDataset'
input_shape = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img'),
    dict(
        type='LoadMask',
        mask_mode='bbox',
        mask_config=dict(
            max_bbox_shape=(128, 128),
            max_bbox_delta=40,
            min_margin=20,
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
        keys=['gt_img', 'masked_img', 'mask', 'mask_bbox'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask']),
    dict(type='ToTensor', keys=['mask_bbox'])
]

test_pipeline = train_pipeline

data_root = 'data/places365'

data = dict(
    workers_per_gpu=4,
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
        test_mode=True))

optimizers = dict(
    generator=dict(type='Adam', lr=0.0004), disc=dict(type='Adam', lr=0.0004))

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=50000)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])

visual_config = dict(
    type='MMEditVisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=[
        'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
    ],
)

evaluation = dict(interval=50000)

total_iters = 500002
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 10000)]
exp_name = 'gl_places'
find_unused_parameters = False
