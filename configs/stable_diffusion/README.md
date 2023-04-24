# Stable Diffusion (2022)

> [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Stable Diffusion is a latent diffusion model conditioned on the text embeddings of a CLIP text encoder, which allows you to create images from text inputs. This model builds upon the CVPR'22 work [High-Resolution Image Synthesis with Latent Diffusion Models](https://ommer-lab.com/research/latent-diffusion-models/). The official code was released at [stable-diffusion](https://github.com/CompVis/stable-diffusion) and also implemented at [diffusers](https://github.com/huggingface/diffusers). We support this algorithm here to facilitate the community to learn together and compare it with other text2image methods.

<!-- [IMAGE] -->

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/12782558/209609229-8221c7cc-d5c9-44d5-a1af-c254b5a95fae.png" width="400"/>
  <br/>
  <b>A mecha robot in a favela in expressionist style</b>
</div></td>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/12782558/210951970-a81e80c3-822e-4782-901e-db52e34b85a3.png" width="400"/>
  <br/>
  <b>A Chinese palace is beside a beautiful lake</b>
</div></td>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/12782558/210952108-df82e5ad-6eb6-4948-8d22-3802299d1131.png" width="400"/>
  <br/>
  <b>A panda is having dinner in KFC</b>
</div></td>
  </tr>
</thead>
</table>

## Pretrained models

|                               Model                               | Dataset | Download |
| :---------------------------------------------------------------: | :-----: | :------: |
| [stable_diffusion_v1.5](./stable-diffusion_ddim_denoisingunet.py) |    -    |    -     |

We use stable diffusion v1.5 weights. This model has several weights including vae, unet and clip.

You may download the weights from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and change the 'from_pretrained' in config to the weights dir.

Download with git:

```shell
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

## Quick Start

Running the following codes, you can get a text-generated image.

```python
from mmengine import MODELS, Config
from torchvision import utils

from mmengine.registry import init_default_scope

init_default_scope('mmagic')

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
config = Config.fromfile(config).copy()
# change the 'pretrained_model_path' if you have downloaded the weights manually
# config.model.unet.from_pretrained = '/path/to/your/stable-diffusion-v1-5'
# config.model.vae.from_pretrained = '/path/to/your/stable-diffusion-v1-5'

StableDiffuser = MODELS.build(config.model)
prompt = 'A mecha robot in a favela in expressionist style'
StableDiffuser = StableDiffuser.to('cuda')

image = StableDiffuser.infer(prompt)['samples'][0]
image.save('robot.png')
```

## Comments

Our codebase for the stable diffusion models builds heavily on [diffusers codebase](https://github.com/huggingface/diffusers) and the model weights are from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

Thanks for the efforts of the community!

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
