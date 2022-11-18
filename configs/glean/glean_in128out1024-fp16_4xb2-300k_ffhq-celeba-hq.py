_base_ = './glean_in128out1024_4xb2-300k_ffhq-celeba-hq.py'

experiment_name = 'glean_in128out1024-fp16_4xb2-300k_ffhq-celeba-hq'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# model settings
model = dict(
    generator=dict(fp16_enabled=True), discriminator=dict(fp16_enabled=True))
