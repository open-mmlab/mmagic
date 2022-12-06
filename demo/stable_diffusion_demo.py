from mmengine import MODELS, Config
from mmedit.utils import register_all_modules
from mmengine.runner import set_random_seed

set_random_seed(1)

register_all_modules()

config = 'configs/stable_diffusion/base_config.py'
StableDiffuser = MODELS.build(Config.fromfile(config).model)
prompt = "clouds surround the mountains and Chinese palaces,sunshine,lake,overlook,overlook,unreal engine,light effect,Dream"
StableDiffuser = StableDiffuser.to("cuda")

image = StableDiffuser.infer(prompt).images[0]
image.save('resources/output/mmedit_sd_cloud_palace.png')
