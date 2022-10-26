_base_ = './glean_x16_2xb8_ffhq.py'

experiment_name = 'glean_x16-fp16_2xb8_ffhq'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# model settings
model = dict(
    generator=dict(fp16_enabled=True), discriminator=dict(fp16_enabled=True))
