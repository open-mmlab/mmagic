_base_ = ['swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py']

experiment_name = 'swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4

# model settings
model = dict(generator=dict(upscale=scale))
