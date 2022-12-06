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
    type='UNet2DConditionModel'
)

vae = dict(
    type='AutoencoderKL'
)
        # num_train_timesteps=1000,
        # beta_start=0.0001,
        # beta_end=0.02,
        # beta_schedule='linear',
        # variance_type='learned_range',
        # timestep_values=None,
        # clip_sample=True,
        # set_alpha_to_one=True,

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


model = dict(
    type='StableDiffuser',
    pretrained_model_name_or_path='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5',
    diffusion_scheduler=diffusion_scheduler,
)