_base_ = './base_tof_vfi_nobn_1xb1_vimeo90k_triplet.py'

experiment_name = 'tof_vfi_spynet_clean_nobn_1xb1_vimeo90k'
work_dir = f'./work_dirs/{experiment_name}'

# pretrained SPyNet
source = 'https://download.openmmlab.com/mmediting/video_interpolators/toflow'
spynet_file = 'pretrained_spynet_sintel_clean_20220321-0756630b.pth'
load_pretrained_spynet = f'{source}/{spynet_file}'

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
        type='EditDataPreprocessor',
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        input_view=(1, -1, 1, 1),
        output_view=(-1, 1, 1),
        pad_size_divisor=16,
        pad_args=dict(mode='reflect')))
