stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
unet = dict(
    type='UNet2DConditionModel',
    subfolder='unet',
    from_pretrained=stable_diffusion_v15_url)
vae = dict(
    type='AutoencoderKL',
    from_pretrained=stable_diffusion_v15_url,
    subfolder='vae')

diffusion_scheduler = dict(
    type='DDPMScheduler',
    from_pretrained=stable_diffusion_v15_url,
    subfolder='scheduler')

model = dict(
    type='StableDiffusion',
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
    test_scheduler=diffusion_scheduler,
    data_preprocessor=dict(type='DataPreprocessor', data_keys=None))
