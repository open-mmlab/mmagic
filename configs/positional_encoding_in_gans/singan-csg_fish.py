_base_ = ['../singan/singan_fish.py']

# MODEL
# NOTE: add by user, e.g.:
# test_pkl_data = './work_dirs/singan_pkl/singan_csg_fis_20210406_175532-f0ec7b61.pkl'  # noqa
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
