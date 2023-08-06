# 0_Deploy_MFNR/0_SeeBetter/configs/controlnet/controlnet-canny.py
# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
controlnet_canny_url = 'lllyasviel/sd-controlnet-canny'

model = dict(
    type='ControlStableDiffusion',
    # vae=dict(type='AutoencoderKL', sample_size=64),
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    # unet=dict(
    #     sample_size=64,
    #     type='UNet2DConditionModel',
    #     down_block_types=('DownBlock2D', ),
    #     up_block_types=('UpBlock2D', ),
    #     block_out_channels=(32, ),
    #     cross_attention_dim=16,
    # ),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=stable_diffusion_v15_url),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    controlnet=dict(
        type='ControlNetModel', from_pretrained=controlnet_canny_url),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor'),
    init_cfg=dict(type='init_from_unet'))
