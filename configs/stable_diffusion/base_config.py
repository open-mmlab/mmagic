feature_extractor = dict(
    type='CLIPFeatureExtractor',
    crop_size=224,
    do_center_crop=True,
    do_convert_rgb=True,
    do_normalize=True,
    do_resize=True,
    feature_extractor_type="CLIPFeatureExtractor",
    image_mean=[
        0.48145466,
        0.4578275,
        0.40821073
    ],
    image_std=[
        0.26862954,
        0.26130258,
        0.27577711
    ],
    resample=3,
    size=224
)

safety_checker = dict(
    type='StableDiffusionSafetyChecker'
)

scheduler = dict(
    type='PNDMScheduler',
    beta_end=0.012,
    beta_schedule="scaled_linear",
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    skip_prk_steps=True,
    steps_offset=1,
    trained_betas=None,
    clip_sample=False
)

text_encoder = dict(
    type='CLIPTextModel',
    attention_dropout=0.0,
    bos_token_id=0,
    dropout=0.0,
    eos_token_id=2,
    hidden_act="quick_gelu",
    hidden_size=768,
    initializer_factor=1.0,
    initializer_range=0.02,
    intermediate_size=3072,
    layer_norm_eps=1e-05,
    max_position_embeddings=77,
    model_type="clip_text_model",
    num_attention_heads=12,
    num_hidden_layers=12,
    pad_token_id=1,
    projection_dim=768,
    torch_dtype="float32",
    transformers_version="4.22.0.dev0",
    vocab_size=49408
)

tokenizer = dict(
    type='CLIPTokenizer'
)

unet = dict(
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
    sample_size=64,
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

pretrained_ckpt_path = dict(
    unet='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin',
    vae='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5/vae/diffusion_pytorch_model.bin'
)

model = dict(
    type='StableDiffuser',
    pretrained_model_name_or_path='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5',
    diffusion_scheduler=diffusion_scheduler,
    unet_cfg=unet,
    vae_cfg=vae,
    pretrained_ckpt_path=pretrained_ckpt_path,
)