unet = dict(
    type='DenoisingUnet',
    image_size=512,
    base_channels=320,
    channels_cfg=[1, 2, 4, 4],
    unet_type='stable',
    act_cfg=dict(type='silu', inplace=False),
    cross_attention_dim=768,
    num_heads=8,
    in_channels=4,
    layers_per_block=2,
    down_block_types=[
        'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
        'DownBlock2D'
    ],
    up_block_types=[
        'UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D',
        'CrossAttnUpBlock2D'
    ],
    output_cfg=dict(var='fixed'))

vae = dict(
    act_fn='silu',
    block_out_channels=[128, 256, 512, 512],
    down_block_types=[
        'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D',
        'DownEncoderBlock2D'
    ],
    in_channels=3,
    latent_channels=4,
    layers_per_block=2,
    norm_num_groups=32,
    out_channels=3,
    sample_size=512,
    up_block_types=[
        'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
        'UpDecoderBlock2D'
    ])

diffusion_scheduler = dict(
    type='DDIMScheduler',
    variance_type='learned_range',
    beta_end=0.012,
    beta_schedule='scaled_linear',
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    clip_sample=False)

tokenizer_path = dict(
    subdir_name='tokenizer',
    merges='merges.txt',
    special_tokens_map='special_tokens_map.json',
    tokenizer_config='tokenizer_config.json',
    vocab='vocab.json')

text_encoder_path = dict(
    subdir_name='text_encoder',
    config='config.json',
    pytorch_model='pytorch_model.bin')

feature_extractor_path = dict(
    subdir_name='feature_extractor', config='preprocessor_config.json')

safety_checker_path = dict(
    subdir_name='safety_checker',
    config='config.json',
    pytorch_model='pytorch_model.bin')

# yapf: disable
pretrained_ckpt_path = dict(
    unet='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin',  # noqa
    vae='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5/vae/diffusion_pytorch_model.bin',  # noqa
    tokenizer=tokenizer_path,
    text_encoder=text_encoder_path,
    feature_extractor=feature_extractor_path,
    safety_checker=safety_checker_path
    )
# yapf: enable

model = dict(
    type='StableDiffusion',
    diffusion_scheduler=diffusion_scheduler,
    unet_cfg=unet,
    vae_cfg=vae,
    pretrained_ckpt_path=pretrained_ckpt_path,
)
