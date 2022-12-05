from mmengine import MODELS, Config
from mmedit.utils import register_all_modules
from mmedit.models.editors.stable_diffusion.utils.training_utils import set_seed

set_seed(1)

register_all_modules()

config = 'configs/stable_diffusion/base_config.py'
StableDiffuser = MODELS.build(Config.fromfile(config).model)
prompt = "clouds surround the mountains and Chinese palaces,sunshine,lake,overlook,overlook,unreal engine,light effect,Dream"
StableDiffuser = StableDiffuser.to("cuda")

image = StableDiffuser.infer(prompt).images[0]
image.save('resources/output/mmedit_sd_cloud_palace.png')
