# define GAN model
model = dict(
    type='SNGAN',
    num_classes=1000,
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(type='SNGANGenerator', output_scale=128, base_channels=64),
    discriminator=dict(
        type='ProjDiscriminator', input_scale=128, base_channels=64),
    discriminator_steps=2)
