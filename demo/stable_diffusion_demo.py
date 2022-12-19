# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import MODELS, Config
from mmedit.utils import register_all_modules
from mmengine.runner import set_random_seed
from torchvision.utils import save_image
import mmcv

# import cv2
# asd = cv2.imread('/nvme/liuwenran/branches/liuwenran/mmediting/resources/output/robot.png')

# asd = cv2.cvtColor(asd, cv2.COLOR_BGR2RGB)
# mmcv.imwrite(asd, 'resources/output/robot_bgr.png')
# import pdb;pdb.set_trace();

set_random_seed(1)

register_all_modules()

# config = 'configs/stable_diffusion/base_config.py'
config = 'configs/stable_diffusion/base_config_denoisingunet.py'
StableDiffuser = MODELS.build(Config.fromfile(config).model)
prompt = "clouds surround the mountains and Chinese palaces,sunshine,lake,overlook,overlook,unreal engine,light effect,Dream"
# prompt = "A beautiful girl with smiling face."
# prompt = "China has won the championship of FIFA."
# prompt = "A man is naked and has much muscle and big penis, big penis"
# prompt = 'A mecha robot in a favela in expressionist style'
# prompt = 'A pikachu fine dining with a view to the Eiffel Tower'
prompt = 'an insect robot preparing a delicious meal'
StableDiffuser = StableDiffuser.to("cuda")

image = StableDiffuser.infer(prompt)['samples']
# image.save('resources/output/insect_sd.png')
mmcv.imwrite(image, 'resources/output/insect_sd.png')
# save_image(image, 'resources/output/insect_sd.png', normalize=True)
# image.save('resources/output/china_champion.png')
# image.save('resources/output/stable_diffusion_nsfw.png')
