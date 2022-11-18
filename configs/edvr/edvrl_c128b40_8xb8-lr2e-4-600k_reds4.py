_base_ = '../_base_/models/base_edvr.py'

experiment_name = 'edvrl_c128b40_8xb8-lr2e-4-600k_reds4'
save_dir = './work_dirs'
work_dir = f'./work_dirs/{experiment_name}'

# load_from = 'https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4_20211228-d895a769.pth'  # noqa: E501

# model settings
model = dict(
    type='EDVR',
    generator=dict(
        type='EDVRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=128,
        num_frames=5,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=40,
        center_frame_idx=2,
        with_tsa=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'),
    train_cfg=dict(tsa_iter=5000),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        input_view=(1, -1, 1, 1),
        output_view=(-1, 1, 1),
    ))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[150000, 150000, 150000, 150000],
    restart_weights=[1, 0.5, 0.5, 0.5],
    eta_min=1e-7)

find_unused_parameters = True
