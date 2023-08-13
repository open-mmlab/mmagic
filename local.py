from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler

unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
unet.save_pretrained("local_unet")

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
vae.save_pretrained("local_vae")

sch = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
sch.save_pretrained("local_scheduler")