model = dict(
    type='StyleGANv1',
    data_preprocessor=dict(type='DataPreprocessor'),
    style_channels=512,
    generator=dict(
        type='StyleGANv1Generator', out_size=None, style_channels=512),
    discriminator=dict(type='StyleGAN1Discriminator', in_size=None))
