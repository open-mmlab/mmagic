_base_ = ['swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py']

experiment_name = 'swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    generator=dict(
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        resi_connection='3conv'))
