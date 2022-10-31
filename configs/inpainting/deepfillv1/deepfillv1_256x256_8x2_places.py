model = dict(
    type='DeepFillv1Inpaintor',
    encdec=dict(
        type='DeepFillEncoderDecoder',
        stage1=dict(
            type='GLEncoderDecoder',
            encoder=dict(type='DeepFillEncoder', padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                in_channels=128,
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=128,
                act_cfg=dict(type='ELU'),
                padding_mode='reflect')),
        stage2=dict(
            type='DeepFillRefiner',
            encoder_attention=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_attention',
                padding_mode='reflect'),
            encoder_conv=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_conv',
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=128,
                act_cfg=dict(type='ELU'),
                padding_mode='reflect'),
            contextual_attention=dict(
                type='ContextualAttentionNeck',
                in_channels=128,
                padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                in_channels=256,
                padding_mode='reflect'))),
    disc=dict(
        type='DeepFillv1Discriminators',
        global_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=256,
            fc_in_channels=256 * 16 * 16,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
        local_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 8 * 8,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2))),
    stage1_loss_type=('loss_l1_hole', 'loss_l1_valid'),
    stage2_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_gan'),
    loss_gan=dict(
        type='GANLoss',
        gan_type='wgan',
        loss_weight=0.0001,
    ),
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    loss_gp=dict(type='GradientPenaltyLoss', loss_weight=10.),
    loss_disc_shift=dict(type='DiscShiftLoss', loss_weight=0.001),
    pretrained=None)

train_cfg = dict(disc_step=5, local_size=(128, 128))
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

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
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
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

optimizers = dict(
    generator=dict(type='Adam', lr=0.0001), disc=dict(type='Adam', lr=0.0001))

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=250000)
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
    interval=10000,
    res_name_list=[
        'gt_img', 'masked_img', 'stage1_fake_res', 'stage1_fake_img',
        'stage2_fake_res', 'stage2_fake_img', 'fake_gt_local'
    ],
)

evaluation = dict(interval=250000)

total_iters = 5000003
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/test_pggan'
load_from = None
resume_from = None
workflow = [('train', 10000)]
exp_name = 'deepfillv1_256x256_8x2_places'
find_unused_parameters = False
