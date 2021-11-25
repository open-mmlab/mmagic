model = dict(
    type='AOTInpaintor',
    encdec=dict(
        type='AOTEncoderDecoder',
        encoder=dict(type='AOTEncoder'),
        decoder=dict(type='AOTDecoder'),
        dilation_neck=dict(
            type='AOTBlockNeck', 
            dilation_rates='1+2+4+8',
            num_aotblock=8)),
    disc=dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True,
        ),
    loss_gan=dict(
        type='SMGANLoss',
        loss_weight=0.01,
    ),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        vgg_type='vgg19',
        layer_weights_perceptual={
            '0': 1.,
            '1': 1.,
            '4': 1.,
            '5': 1.,
            '6': 1.,
            '9': 1.,
            '10': 1.,
            '11': 1.,
            '18': 1.,
            '19': 1.,
            '20': 1.,
            '27': 1.,
            '28': 1.,
            '29': 1.,
        },
        layer_weights_style={
            '7': 1.,
            '8': 1.,
            '16': 1.,
            '17': 1.,
            '25': 1.,
            '26': 1.,
            '30': 1.,
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
    dict(type='LoadMask',
        mask_mode='set',
        mask_config=dict(
            mask_list_file='/home/SENSETIME/lintsuihin/code/mmediting/data/places365/mask_list.txt',
            prefix='/home/SENSETIME/lintsuihin/code/mmediting/data/places365/mask_256/',
            io_backend='disk',
            flag='unchanged',
            file_client_kwargs=dict())),
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

data_root = 'data/places365_test'

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
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
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9)), disc=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9)))

lr_config = dict(policy='Fixed', by_epoch=False)

checkpoint_config = dict(by_epoch=False, interval=50000)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])

visual_config = dict(
    type='VisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=[
        'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
    ],
)

# evaluation = dict(interval=50000)

total_iters = 500002
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './workdirs'
load_from = None
resume_from = None
workflow = [('train', 10000)]
exp_name = 'gl_places'
find_unused_parameters = False
