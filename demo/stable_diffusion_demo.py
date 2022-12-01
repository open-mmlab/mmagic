from mmengine import Config, MODELS
from mmedit.utils import register_all_modules
register_all_modules()
cfg = dict(
        type='StableDiffuser',
        pretrained_model_name_or_path='/nvme/liuwenran/repos/diffusers/resources/stable-diffusion-v1-5',
        class_type='StableDiffusionPipeline'
    )

StableDiffuser = MODELS.build(cfg)
prompt = "clouds surround the mountains and Chinese palaces,sunshine,lake,overlook,overlook,unreal engine,light effect,Dream"
image = StableDiffuser.infer(prompt)
image.save('resources/output/mmedit_sd_cloud_palace.png')
