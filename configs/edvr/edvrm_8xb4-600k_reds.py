_base_ = '../_base_/models/base_edvr.py'

experiment_name = 'edvrm_8xb4-600k_reds'
save_dir = './work_dirs'
work_dir = f'./work_dirs/{experiment_name}'

# model settings
model = dict(
    type='EDVR',
    generator=dict(
        type='EDVRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_frames=5,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=10,
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

train_dataloader = dict(num_workers=6, batch_size=4)

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[50000, 100000, 150000, 150000, 150000],
    restart_weights=[1, 1, 1, 1, 1],
    eta_min=1e-7)

find_unused_parameters = True
