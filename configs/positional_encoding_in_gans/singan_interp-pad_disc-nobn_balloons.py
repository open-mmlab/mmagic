_base_ = ['../singan/singan_balloons.py']

# TODO: have bugs
# MODEL
# NOTE: add by user, e.g.:
# test_pkl_data = './work_dirs/singan_pkl/singan_interp-pad_disc-nobn_balloons_20210406_180059-7d63e65d.pkl'  # noqa
test_pkl_data = None

model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE', interp_pad=True, noise_with_pad=True),
    discriminator=dict(norm_cfg=None),
    fixed_noise_with_pad=True)
