model = dict(
    type='AOTInpaintor',
    encdec=dict(
        type='AOTEncoderDecoder',
        encoder=dict(type='AOTEncoder'),
        decoder=dict(type='AOTDecoder'),
        dilation_neck=dict(
            type='AOTBlockNeck', dilation_rates=(1, 2, 4, 8), num_aotblock=8)),
    disc=dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True,
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='smgan',
        loss_weight=0.01,
    ),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        vgg_type='vgg19',
        layer_weights={
            '1': 1.,
            '6': 1.,
            '11': 1.,
            '20': 1.,
            '29': 1.,
        },
        layer_weights_style={
            '8': 1.,
            '17': 1.,
            '26': 1.,
            '31': 1.,
        },
        perceptual_weight=0.1,
        style_weight=250),
    loss_out_percep=True,
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.,
    ),
    pretrained=None)

train_cfg = dict(disc_step=1)
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

dataset_type = 'ImgInpaintingDataset'
input_shape = (512, 512)

mask_root = 'data/masks'

train_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img', channel_order='rgb'),
    dict(
        type='LoadMask',
        mask_mode='set',
        mask_config=dict(
            mask_list_file=f'{mask_root}/train_places_mask_list.txt',
            prefix=mask_root,
            io_backend='disk',
            flag='unchanged',
            file_client_kwargs=dict())),
    dict(
        type='RandomResizedCrop',
        keys=['gt_img'],
        crop_size=input_shape,
    ),
    dict(type='Flip', keys=['gt_img', 'mask'], direction='horizontal'),
    dict(
        type='Resize',
        keys=['mask'],
        scale=input_shape,
        keep_ratio=False,
        interpolation='nearest'),
    dict(type='RandomRotation', keys=['mask'], degrees=(0.0, 45.0)),
    dict(
        type='ColorJitter',
        keys=['gt_img'],
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
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

test_pipeline = [
    dict(type='LoadImageFromFile', key='gt_img', channel_order='rgb'),
    dict(
        type='LoadMask',
        mask_mode='set',
        mask_config=dict(
            mask_list_file=f'{mask_root}/mask_0.5-0.6_list.txt',
            prefix=mask_root + '/mask_512',
            io_backend='disk',
            flag='unchanged',
            file_client_kwargs=dict())),
    dict(
        type='RandomResizedCrop',
        keys=['gt_img'],
        crop_size=(512, 512),
    ),
    dict(
        type='Normalize',
        keys=['gt_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=True),
    dict(type='GetMaskedImage'),
    dict(
        type='Collect',
        keys=['gt_img', 'masked_img', 'mask'],
        meta_keys=['gt_img_path']),
    dict(type='ImageToTensor', keys=['gt_img', 'masked_img', 'mask'])
]

data_root = 'data/places365'

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=12, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/train_places_img_list.txt',
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
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9)),
    disc=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9)))

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=10000)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])

visual_config = dict(
    type='MMEditVisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=['gt_img', 'masked_img', 'fake_res', 'fake_img'],
)

evaluation = dict(interval=50000)

total_iters = 500002
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './workdirs/aotgan_places'
load_from = None
resume_from = None
workflow = [('train', 10000)]
exp_name = 'AOT-GAN_512x512_4x12_places'
find_unused_parameters = False
