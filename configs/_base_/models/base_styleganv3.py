# define GAN model

d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

model = dict(
    type='StyleGAN3',
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(
        type='StyleGANv3Generator',
        noise_size=512,
        style_channels=512,
        out_size=None,  # Need to be set.
        img_channels=3,
    ),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=None,  # Need to be set.
    ),
    ema_config=dict(type='ExponentialMovingAverage'),
    loss_config=dict(
        r1_loss_weight=10. / 2. * d_reg_interval,
        r1_interval=d_reg_interval,
        norm_mode='HWC',
        g_reg_interval=g_reg_interval,
        g_reg_weight=2. * g_reg_interval,
        pl_batch_shrink=2))
