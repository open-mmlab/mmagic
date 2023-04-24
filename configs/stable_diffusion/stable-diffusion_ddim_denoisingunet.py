unet = dict(
    type='DenoisingUnet',
    image_size=512,
    base_channels=320,
    channels_cfg=[1, 2, 4, 4],
    unet_type='stable',
    act_cfg=dict(type='silu'),
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
    type='EditAutoencoderKL',
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
    type='EditDDIMScheduler',
    variance_type='learned_range',
    beta_end=0.012,
    beta_schedule='scaled_linear',
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    clip_sample=False)

model = dict(
    type='StableDiffusion',
    unet=unet,
    vae=vae,
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
        subfolder='text_encoder'),
    tokenizer='runwayml/stable-diffusion-v1-5',
    scheduler=diffusion_scheduler,
    test_scheduler=diffusion_scheduler,
    tomesd_cfg=dict(ratio=0.5),
    init_cfg=dict())
