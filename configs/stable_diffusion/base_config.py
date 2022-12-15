unet = dict(
    type="UNet2DConditionModel",
    act_fn="silu",
    attention_head_dim=8,
    block_out_channels=[
        320,
        640,
        1280,
        1280
    ],
    center_input_sample=False,
    cross_attention_dim=768,
    down_block_types=[
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    downsample_padding=1,
    flip_sin_to_cos=True,
    freq_shift=0,
    in_channels=4,
    layers_per_block=2,
    mid_block_scale_factor=1,
    norm_eps=1e-05,
    norm_num_groups=32,
    out_channels=4,
    up_block_types=[
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ]
)

vae = dict(
    act_fn="silu",
    block_out_channels=[
        128,
        256,
        512,
        512
    ],
    down_block_types=[
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D"
    ],
    in_channels=3,
    latent_channels=4,
    layers_per_block=2,
    norm_num_groups=32,
    out_channels=3,
    sample_size=512,
    up_block_types=[
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"
    ]
)

diffusion_scheduler = dict(
    type='DDIMScheduler',
    variance_type='learned_range',
    beta_end=0.012,
    beta_schedule="scaled_linear",
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    clip_sample=False
)

tokenizer_path = dict(
    subdir_name='tokenizer',
    merges='merges.txt',
    special_tokens_map='special_tokens_map.json',
    tokenizer_config='tokenizer_config.json',
    vocab='vocab.json'
)

text_encoder_path = dict(
    subdir_name='text_encoder',
    config='config.json',
    pytorch_model='pytorch_model.bin'
)

feature_extractor_path = dict(
    subdir_name='feature_extractor',
    config='preprocessor_config.json'
)

safety_checker_path = dict(
    subdir_name='safety_checker',
    config='config.json',
    pytorch_model='pytorch_model.bin'
)

pretrained_ckpt_path = dict(
    unet='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin',
    vae='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5/vae/diffusion_pytorch_model.bin',
    tokenizer=tokenizer_path,
    text_encoder=text_encoder_path,
    feature_extractor=feature_extractor_path,
    safety_checker=safety_checker_path
)

model = dict(
    type='StableDiffuser',
    diffusion_scheduler=diffusion_scheduler,
    unet_cfg=unet,
    vae_cfg=vae,
    pretrained_ckpt_path=pretrained_ckpt_path,
)