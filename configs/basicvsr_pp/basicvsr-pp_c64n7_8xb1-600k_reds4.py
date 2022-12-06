_base_ = '../basicvsr/basicvsr_2xb4_reds4.py'

experiment_name = 'basicvsr-pp_c64n7_8xb1-600k_reds4'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        input_view=(1, -1, 1, 1),
        output_view=(1, -1, 1, 1),
    ))

train_dataloader = dict(
    num_workers=6, batch_size=1, dataset=dict(num_input_frames=30))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=600_000, val_interval=5000)

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}))

default_hooks = dict(checkpoint=dict(out_dir=save_dir))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[600000],
    restart_weights=[1],
    eta_min=1e-7)
