# define GAN model
model = dict(
    type='SNGAN',
    num_classes=10,
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(type='SNGANGenerator', output_scale=32, base_channels=256),
    discriminator=dict(
        type='ProjDiscriminator', input_scale=32, base_channels=128),
    discriminator_steps=5)
