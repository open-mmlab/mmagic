_base_ = '../__base__/models/base_liif_c64b16_g1_1000k_div2k.py'

experiment_name = 'liif_rdn_norm_x2-4_c64b16_g1_1000k_div2k'
work_dir = f'./work_dirs/{experiment_name}'

scale_min, scale_max = 1, 4
scale_test = 4

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
        type='EditDataPreprocessor',
        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        input_view=(-1, 1, 1),
        output_view=(1, -1)))
