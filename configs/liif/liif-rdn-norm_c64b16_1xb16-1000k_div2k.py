_base_ = [
    '../_base_/models/base_liif.py', '../_base_/datasets/liif_test_config.py'
]

experiment_name = 'liif-rdn-norm_c64b16_1xb16-1000k_div2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale_min, scale_max = 1, 4

# model settings
model = dict(
    type='LIIF',
    generator=dict(
        type='LIIFRDNNet',
        encoder=dict(
            type='RDNNet',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=4,
            num_layers=8,
            channel_growth=64),
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
        type='DataPreprocessor',
        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    ))
