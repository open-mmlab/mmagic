_base_ = '../basicvsr/basicvsr_vimeo90k_bi.py'

experiment_name = 'basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi'
work_dir = f'./work_dirs/{experiment_name}'

load_from = 'https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth'  # noqa

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=-1),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        input_view=(1, -1, 1, 1),
        output_view=(1, -1, 1, 1),
    ))

train_dataloader = dict(num_workers=6, batch_size=2)

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    optimizer=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}))

default_hooks = dict(
    checkpoint=dict(out_dir='sh1984:s3://ysli/basicvsr_plusplus'))
