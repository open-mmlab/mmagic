_base_ = './glean_x8_2xb8_cat.py'

experiment_name = 'glean_x8-fp16_2xb8_cat'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

# model settings
model = dict(
    generator=dict(fp16_enabled=True), discriminator=dict(fp16_enabled=True))
