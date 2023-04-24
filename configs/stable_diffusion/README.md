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

|                               Model                               | Dataset |                            Download                            |
| :---------------------------------------------------------------: | :-----: | :------------------------------------------------------------: |
| [stable_diffusion_v1.5](./stable-diffusion_ddim_denoisingunet.py) |    -    | [model](https://huggingface.co/runwayml/stable-diffusion-v1-5) |

We use stable diffusion v1.5 weights. This model has several weights including vae, unet and clip.

You should download the weights from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and change the 'pretrained_model_path' in config to the weights dir.

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

init_default_scope('mmedit')

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
config = Config.fromfile(config).copy()
config.model.init_cfg.pretrained_model_path = '/path/to/your/stable-diffusion-v1-5'

StableDiffuser = MODELS.build(config.model)
prompt = 'A mecha robot in a favela in expressionist style'
StableDiffuser = StableDiffuser.to('cuda')

image = StableDiffuser.infer(prompt)['samples']
image[0].save('robot.png')
```

## Use ToMe to accelerate your stable diffusion model

We support **[tomesd](https://github.com/dbolya/tomesd)** now! It is developed based on [ToMe](https://github.com/facebookresearch/ToMe), an efficient ViT speed-up tool based on token merging. To work on with **tomesd** in `mmediting`, you just need to add `tomesd_cfg` to `model` in [stable-diffusion-config](configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py).

```python
...
model = dict(
    type='StableDiffusion',
    unet=unet,
    vae=vae,
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
        subfolder='text_encoder'),
    tokenizer='runwayml/stable-diffusion-v1-5',
    scheduler=diffusion_scheduler,
    test_scheduler=diffusion_scheduler,
    tomesd_cfg=dict(
        ratio=0.5),
    init_cfg=dict())
```

Then following the code below, you can evaluate the speed-up performance on stable diffusion model.

```python
import time
import numpy as np

from mmengine import MODELS, Config
from mmengine.registry import init_default_scope

init_default_scope('mmedit')

size = 512
ratios = [0.5, 0.75]
samples_perprompt = 5

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
config = Config.fromfile(config).copy()
config.model.init_cfg.pretrained_model_path = '/path/to/your/stable-diffusion-v1-5'

prompt = 'A mecha robot in a favela in expressionist style'

for ratio in ratios:
    # use tomesd
    config.model.tomesd_cfg.ratio = ratio
    # # do not use tomesd
    # config.model.tomesd_cfg = None

    sd_model = MODELS.build(config.model).to('cuda')

    t = time.time()
    for i in range(100//samples_perprompt):
        image = sd_model.infer(prompt, height=size, width=size, num_images_per_prompt=samples_perprompt)['samples']

    print(f"Generating 100 images with {samples_perprompt} images per prompt, merging ratio {ratio}, time used : {time.time() - t}s")

# # Results
# | `ratio`  | size | num_images_per_prompt |       time (s)   |
# | w/o tome | 512  |           5           |       578.86     |
# |   0.5    | 512  |           5           |  436.45 (↓24.6%) |
# |   0.75   | 512  |           5           |  389.77 (↓32.7%) |
```

The detailed settings for **tomesd_cfg** are as follows:

- `ratio` **(float)**: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.The maximum value for this is 1-(1/(`sx` * `sy`)). **By default, the max ratio is 0.75, usually \<= 0.5 is recommended.** Higher values result in more speed-up, but with more visual quality loss.
- `max_downsample` **(int)**: Apply ToMe to layers with at most this amount of downsampling. E.g., 1 only applies to layers with no downsampling, while 8 applies to all layers. Should be chosen from $1, 2, 4, 8$. **1 and 2 are recommended.**
- `sx`, `sy` **(int, int)**: The stride for computing dst sets. A higher stride means you can merge more tokens, **default setting of (2, 2) works well in most cases**. `sx` and `sy` do not need to divide image size.
- `use_rand` **(bool)**: Whether or not to allow random perturbations when computing dst sets. By default: True, but if you're having weird artifacts you can try turning this off.
- `merge_attn` **(bool)**: Whether or not to merge tokens for attention **(recommended)**.
- `merge_crossattn` **(bool)**: Whether or not to merge tokens for cross attention **(not recommended)**.
- `merge_mlp` **(bool)**: Whether or not to merge tokens for the mlp layers **(particular not recommended)**.

For more details about the **tomesd** setting, please refer to [Token Merging for Stable Diffusion](https://arxiv.org/abs/2303.17604).

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

@article{bolya2023tomesd,
  title={Token Merging for Fast Stable Diffusion},
  author={Bolya, Daniel and Hoffman, Judy},
  journal={arXiv},
  year={2023}
}

@inproceedings{bolya2023tome,
  title={Token Merging: Your {ViT} but Faster},
  author={Bolya, Daniel and Fu, Cheng-Yang and Dai, Xiaoliang and Zhang, Peizhao and Feichtenhofer, Christoph and Hoffman, Judy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
