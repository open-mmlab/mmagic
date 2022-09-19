_base_ = ['../singan/singan_fish.py']

# MODEL
# NOTE: add by user, e.g.:
# test_pkl_data = './work_dirs/singan_pkl/singan_spe-dim4_fish_20210406_175933-f483a7e3.pkl'  # noqa
test_pkl_data = None

embedding_dim = 4
num_scales = 10  # start from zero
model = dict(
    type='PESinGAN',
    num_scales=num_scales,
    generator=dict(
        type='SinGANMSGeneratorPE',
        num_scales=num_scales,
        padding=1,
        pad_at_head=False,
        first_stage_in_channels=embedding_dim * 2,
        positional_encoding=dict(
            type='SPE',
            embedding_dim=embedding_dim,
            padding_idx=0,
            init_size=512,
            div_half_dim=False,
            center_shift=200)),
    discriminator=dict(num_scales=num_scales),
    test_pkl_data=test_pkl_data)
