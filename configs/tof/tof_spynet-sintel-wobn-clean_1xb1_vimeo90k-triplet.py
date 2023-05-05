_base_ = '../_base_/models/base_tof.py'

experiment_name = 'tof_spynet-sintel-wobn-clean_1xb1_vimeo90k-triplet'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# pretrained SPyNet
load_pretrained_spynet = 'https://download.openmmlab.com/mmediting/video_' +\
    'interpolators/toflow/pretrained_spynet_sintel_clean_20220321-0756630b.pth'

# model settings
model = dict(
    type='BasicInterpolator',
    generator=dict(
        type='TOFlowVFINet',
        flow_cfg=dict(norm_cfg=None, pretrained=load_pretrained_spynet)),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    required_frames=2,
    step_frames=1,
    init_cfg=None,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        pad_size_divisor=16,
        pad_mode='reflect',
    ))
