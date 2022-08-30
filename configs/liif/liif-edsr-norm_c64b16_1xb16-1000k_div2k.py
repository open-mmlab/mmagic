_base_ = [
    '../_base_/models/base_liif.py', '../_base_/datasets/liif_test_config.py'
]

experiment_name = 'liif-edsr-norm_c64b16_1xb16-1000k_div2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale_min, scale_max = 1, 4

# model settings
model = dict(
    type='LIIF',
    generator=dict(
        type='LIIFEDSRNet',
        encoder=dict(
            type='EDSRNet',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16),
        imnet=dict(
            type='MLPRefiner',
            in_dim=64,
            out_dim=3,
            hidden_list=[256, 256, 256, 256]),
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
        eval_bsize=30000),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0.4488 * 255, 0.4371 * 255, 0.4040 * 255],
        std=[255., 255., 255.],
        input_view=(-1, 1, 1),
        output_view=(1, -1)))
