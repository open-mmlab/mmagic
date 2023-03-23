# Controlnet Animation (2022)

> [Controlnet](https://github.com/lllyasviel/ControlNet)

> **Task**: controlnet_animation

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

ControlNet is a neural network structure to control diffusion models by adding extra conditions.
We use controlnet to generate frames guided by input video and make animation.

## Pretrained models

config here

|                Model                 | Dataset | Download |
| :----------------------------------: | :-----: | :------: |
| [anythingv3](./anythingv3_config.py) |    -    |    -     |

## Quick Start

Running the following codes, you can get a text-generated image.

```python
from mmengine import MODELS, Config
from torchvision import utils

from mmengine.registry import init_default_scope

init_default_scope('mmedit')

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
StableDiffuser = MODELS.build(Config.fromfile(config).model)
prompt = 'A mecha robot in a favela in expressionist style'
StableDiffuser = StableDiffuser.to('cuda')

image = StableDiffuser.infer(prompt)['samples']
utils.save_image(image, 'robot.png')
```

## Citation

```bibtex
@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Lvmin Zhang and Maneesh Agrawala},
  year={2023},
  eprint={2302.05543},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
