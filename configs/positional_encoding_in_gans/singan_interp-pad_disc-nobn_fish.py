_base_ = ['../singan/singan_fish.py']

# TODO: have bugs
# MODEL
# test_pkl_data = './work_dirs/singan_pkl/singan_interp-pad_disc-nobn_fis_20210406_175720-9428517a.pkl'  # noqa
test_pkl_data = None

model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE', interp_pad=True, noise_with_pad=True),
    discriminator=dict(norm_cfg=None),
    fixed_noise_with_pad=True,
    test_pkl_data=test_pkl_data)
