feature_extractor = dict(
    type='CLIPFeatureExtractor'
)

safety_checker = dict(
    type='StableDiffusionSafetyChecker'
)

scheduler = dict(
    type='PNDMScheduler'
)

text_encoder = dict(
    type='CLIPTextModel'
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


model = dict(
    type='StableDiffuser',
    pretrained_model_name_or_path='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5',
)