# Stable Diffusion (2022)

> [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

> **Task**: Text2Image, Inpainting

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
  <b>A panda is having dinner at KFC</b>
</div></td>
  </tr>
</thead>
</table>

## Pretrained models

|                                        Model                                         |    Task    | Dataset | Download |
| :----------------------------------------------------------------------------------: | :--------: | :-----: | :------: |
|          [stable_diffusion_v1.5](./stable-diffusion_ddim_denoisingunet.py)           | Text2Image |    -    |    -     |
| [stable_diffusion_v1.5_tomesd](./stable-diffusion_ddim_denoisingunet-tomesd_5e-1.py) | Text2Image |    -    |    -     |
|  [stable_diffusion_v1.5_inpaint](./stable-diffusion_ddim_denoisingunet-inpaint.py)   | Inpainting |    -    |    -     |

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

To inpaint an image, you could run the following codes.

```python
import mmcv
from mmengine import MODELS, Config
from mmengine.registry import init_default_scope
from PIL import Image

init_default_scope('mmagic')

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet-inpaint.py'
config = Config.fromfile(config).copy()
# change the 'pretrained_model_path' if you have downloaded the weights manually
# config.model.unet.from_pretrained = '/path/to/your/stable-diffusion-inpainting'
# config.model.vae.from_pretrained = '/path/to/your/stable-diffusion-inpainting'

StableDiffuser = MODELS.build(config.model)
prompt = 'a mecha robot sitting on a bench'

img_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'  # noqa
mask_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'  # noqa

image = Image.fromarray(mmcv.imread(img_url, channel_order='rgb'))
mask = Image.fromarray(mmcv.imread(mask_url)).convert('L')
StableDiffuser = StableDiffuser.to('cuda')

image = StableDiffuser.infer(
    prompt,
    image,
    mask
)['samples'][0]
image.save('inpaint.png')
```

## Use ToMe to accelerate your stable diffusion model

We support **[tomesd](https://github.com/dbolya/tomesd)** now! It is developed based on [ToMe](https://github.com/facebookresearch/ToMe), an efficient ViT speed-up tool based on token merging. To work on with **tomesd** in `mmagic`, you just need to add `tomesd_cfg` to `model` as shown in [stable_diffusion_v1.5_tomesd](stable-diffusion_ddim_denoisingunet-tomesd_5e-1.py). The only requirement is `torch >= 1.12.1` in order to properly support `torch.Tensor.scatter_reduce()` functionality. Please do check it before running the demo.

```python
...
model = dict(
    type='StableDiffusion',
    unet=unet,
    vae=vae,
    enable_xformers=False,
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=diffusion_scheduler,
    test_scheduler=diffusion_scheduler,
    tomesd_cfg=dict(
        ratio=0.5))
```

The detailed settings for `tomesd_cfg` are as follows:

- `ratio (float)`: The ratio of tokens to merge. For example, 0.4 would reduce the total number of tokens by 40%.The maximum value for this is 1-(1/(`sx` * `sy`)). **By default, the max ratio is 0.75, usually \<= 0.5 is recommended.** Higher values result in more speed-up, but with more visual quality loss.
- `max_downsample (int)`: Apply ToMe to layers with at most this amount of downsampling. E.g., 1 only applies to layers with no downsampling, while 8 applies to all layers. Should be chosen from 1, 2, 4, 8. **1, 2 are recommended.**
- `sx, sy (int, int)`: The stride for computing dst sets. A higher stride means you can merge more tokens, **default setting of (2, 2) works well in most cases**. `sx` and `sy` do not need to divide image size.
- `use_rand (bool)`: Whether or not to allow random perturbations when computing dst sets. By default: True, but if you're having weird artifacts you can try turning this off.
- `merge_attn (bool)`: Whether or not to merge tokens for attention **(recommended)**.
- `merge_crossattn (bool)`: Whether or not to merge tokens for cross attention **(not recommended)**.
- `merge_mlp (bool)`: Whether or not to merge tokens for the mlp layers **(especially not recommended)**.

For more details about the **tomesd** setting, please refer to [Token Merging for Stable Diffusion](https://arxiv.org/abs/2303.17604).

Then following the code below, you can evaluate the speed-up performance on stable diffusion models or stable-diffusion-based models ([DreamBooth](../dreambooth/README.md), [ControlNet](../controlnet/README.md)).

```python
import time
import numpy as np

from mmengine import MODELS, Config
from mmengine.registry import init_default_scope

init_default_scope('mmagic')

_device = 0
work_dir = '/path/to/your/work_dir'
config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet-tomesd_5e-1.py'
config = Config.fromfile(config).copy()
# # change the 'pretrained_model_path' if you have downloaded the weights manually
# config.model.unet.from_pretrained = '/path/to/your/stable-diffusion-v1-5'
# config.model.vae.from_pretrained = '/path/to/your/stable-diffusion-v1-5'

# w/o tomesd
config.model.tomesd_cfg = None
StableDiffuser = MODELS.build(config.model).to(f'cuda:{_device}')
prompt = 'A mecha robot in a favela in expressionist style'

# inference time evaluation params
size = 512
ratios = [0.5, 0.75]
samples_perprompt = 5

t = time.time()
for i in range(100//samples_perprompt):
    image = StableDiffuser.infer(prompt, height=size, width=size, num_images_per_prompt=samples_perprompt)['samples'][0]
    if i == 0:
        image.save(f"{work_dir}/wo_tomesd.png")
print(f"Generating 100 images with {samples_perprompt} images per prompt, without ToMe speed-up, time used : {time.time() - t}s")

for ratio in ratios:
    # w/ tomesd
    config.model.tomesd_cfg = dict(ratio=ratio)
    sd_model = MODELS.build(config.model).to(f'cuda:{_device}')

    t = time.time()
    for i in range(100//samples_perprompt):
        image = sd_model.infer(prompt, height=size, width=size, num_images_per_prompt=samples_perprompt)['samples'][0]
        if i == 0:
            image.save(f"{work_dir}/w_tomesd_ratio_{ratio}.png")

    print(f"Generating 100 images with {samples_perprompt} images per prompt, merging ratio {ratio}, time used : {time.time() - t}s")
```

Here are some inference performance comparisons running on **single RTX 3090** with `torch 2.0.0+cu118` as backends. The results are reasonable, when enabling `xformers`, the speed-up ratio is a little bit lower. But `tomesd` still effectively reduces the inference time. It is especially recommended that enable `tomesd` when the `image_size` and `num_images_per_prompt` are large, since the number of similar tokens are larger and `tomesd` can achieve better performance.

|                              Model                               |    Task    | Dataset | Download | xformer |            Ratio            | Size / Num images per prompt |                     Time (s)                      |
| :--------------------------------------------------------------: | :--------: | :-----: | :------: | :-----: | :-------------------------: | :--------------------------: | :-----------------------------------------------: |
| [stable_diffusion_v1.5-tomesd](./stable-diffusion_ddim_denoisingunet-tomesd_5e-1.py) | Text2Image |    -    |    -     |   w/o   | w/o tome <br> 0.5 <br> 0.75 |         512  /    5          | 542.20 <br> 427.65 (↓21.1%) <br>  393.05 (↓27.5%) |
| [stable_diffusion_v1.5-tomesd](./stable-diffusion_ddim_denoisingunet-tomesd_5e-1.py) | Text2Image |    -    |    -     |   w/    | w/o tome <br> 0.5 <br> 0.75 |         512  /    5          | 541.64 <br> 428.53 (↓20.9%) <br>  396.38 (↓26.8%) |

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/49406546/234613951-43b94470-89ff-4edc-a2e2-bea9e7a8a566.png" width="400"/>
  <br/>
  <b> w/o ToMe </b>
</div></td>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/49406546/234613969-213e3436-b73c-4b8e-82ce-b91492b44db3.png" width="400"/>
  <br/>
  <b> w/ ToMe Speed-up (token merge ratio=0.5) </b>
</div></td>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/49406546/234613983-82fee9a3-05f7-4b1d-85dc-a507e85ecb31.png" width="400"/>
  <br/>
  <b> w/ ToMe Speed-up (token merge ratio=0.75) </b>
</div></td>
  </tr>
</thead>
</table>

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
