# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

model = dict(
    type='PConvInpaintor',
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
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
    ))

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.00005))

# learning policy
# Fixed
