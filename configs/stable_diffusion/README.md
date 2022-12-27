# Stable Diffusion (2022)

> [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Stable Diffusion is a latent diffusion model conditioned on the text embeddings of a CLIP text encoder, which allows you to create images from text inputs.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12782558/209609229-8221c7cc-d5c9-44d5-a1af-c254b5a95fae.png" width="400"/>
</div >

## Results and models

We provide stable diffusion v1.5 weights. This model has several weights including vae, unet and clip. We put the urls in config file and it will be downloaded automatically when you run codes following the [Quick Start](#quick-start)

|    Diffusion Model    |                       Config                       |                Download                |
| :-------------------: | :------------------------------------------------: | :------------------------------------: |
| stable_diffusion_v1.5 | [config](./stable-diffusion_ddim_denoisingunet.py) | Urls could be found in the config file |

## Quick Start

Running the following codes, you can get a text-generated image.

```python
from mmengine import MODELS, Config
from torchvision import utils

from mmedit.utils import register_all_modules

register_all_modules()

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
StableDiffuser = MODELS.build(Config.fromfile(config).model)
prompt = 'A mecha robot in a favela in expressionist style'
StableDiffuser = StableDiffuser.to('cuda')

image = StableDiffuser.infer(prompt)['samples']
utils.save_image(image, 'robot.png')
```

## Comments

Our codebase for the stable diffusion models builds heavily on [diffusers codebase](https://github.com/huggingface/diffusers) and the model weights are from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). Thanks for the efforts of the community!

## Citation

```bibtex
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
