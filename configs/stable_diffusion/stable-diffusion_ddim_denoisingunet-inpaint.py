# Use DiffuserWrapper!
stable_diffusion_v15_url = 'runwayml/stable-diffusion-inpainting'
unet = dict(
    type='UNet2DConditionModel',
    subfolder='unet',
    from_pretrained=stable_diffusion_v15_url)
vae = dict(
    type='AutoencoderKL',
    from_pretrained=stable_diffusion_v15_url,
    subfolder='vae')

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
    type='StableDiffusionInpaint',
    unet=unet,
    vae=vae,
    enable_xformers=False,
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=diffusion_scheduler,
    test_scheduler=diffusion_scheduler)
