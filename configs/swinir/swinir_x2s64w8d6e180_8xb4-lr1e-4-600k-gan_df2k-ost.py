_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/realsr_test_config.py'
]

experiment_name = 'swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 2
img_size = 64

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SwinIRNet',
        upscale=scale,
        in_chans=3,
        img_size=img_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='EditDataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))
