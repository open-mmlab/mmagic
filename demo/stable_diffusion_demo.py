# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import MODELS, Config
from torchvision import utils

from mmedit.utils import register_all_modules

register_all_modules()

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
StableDiffuser = MODELS.build(Config.fromfile(config).model)
# prompt = 'clouds surround the mountains and Chinese palaces,' + \
#             'sunshine,lake,overlook,overlook,unreal engine,' + \
#             'light effect,Dream'
# prompt = "A beautiful girl with smiling face."
# prompt = "China has won the championship of FIFA."
# prompt = "A man is naked and has much muscle and big penis, big penis"
# prompt = 'A mecha robot in a favela in expressionist style'
# prompt = 'A pikachu fine dining with a view to the Eiffel Tower'
prompt = 'an insect robot preparing a delicious meal'
StableDiffuser = StableDiffuser.to('cuda')

image = StableDiffuser.infer(prompt)['samples']
# mmcv.imwrite(image, 'resources/output/insect_sd.png')
utils.save_image(image, 'resources/output/insect_sd.png')
