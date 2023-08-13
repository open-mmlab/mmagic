import torch
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

def process_state_dict(state_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v

    return new_state_dict

cfg = Config.fromfile('configs/vico/vico.py')
checkpoint = torch.load("work_dirs/vico/iter_500.pth")
state_dict = process_state_dict(checkpoint['state_dict'])
vico = MODELS.build(cfg.model)
vico.load_state_dict(state_dict)
vico = vico.cuda()

prompt = ["A photo of S*", "A S* on the grass"]
reference = "data/vico/wooden_pot_debug/1.png"
image_ref = Image.open(reference)
with torch.no_grad():
    output = vico.infer(prompt=prompt, image_reference=image_ref, seed=123)['sample'][0]
output.save("infer.png")