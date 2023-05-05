_base_ = '../basicvsr/basicvsr_2xb4_reds4.py'

experiment_name = 'iconvsr_2xb4_reds4'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='IconVSRNet',
        mid_channels=64,
        num_blocks=30,
        keyframe_stride=5,
        padding=2,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        edvr_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'iconvsr/edvrm_reds_20210413-3867262f.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

default_hooks = dict(checkpoint=dict(out_dir=save_dir))
