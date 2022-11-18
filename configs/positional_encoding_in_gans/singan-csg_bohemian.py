_base_ = ['../singan/singan_bohemian.py']

# MODEL
# NOTE: add by user, e.g.:
# test_pkl_data = './work_dirs/singan_pkl/singan_csg_bohemian_20210407_195455-5ed56db2.pkl'  # noqa
test_pkl_data = None

num_scales = 10  # start from zero
model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE',
        num_scales=num_scales,
        padding=1,
        pad_at_head=False,
        first_stage_in_channels=2,
        positional_encoding=dict(type='CSG')),
    discriminator=dict(num_scales=num_scales),
    first_fixed_noises_ch=2,
    test_pkl_data=test_pkl_data)
