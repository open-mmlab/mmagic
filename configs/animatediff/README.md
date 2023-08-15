# AnimateDiff (2023)

> [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)

> **Task**: Text2Video

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

With the advance of text-to-image models (e.g., Stable Diffusion) and corresponding personalization techniques such as DreamBooth and LoRA, everyone can manifest their imagination into high-quality images at an affordable cost. Subsequently, there is a great demand for image animation techniques to further combine generated static images with motion dynamics. In this report, we propose a practical framework to animate most of the existing personalized text-to-image models once and for all, saving efforts in model-specific tuning. At the core of the proposed framework is to insert a newly initialized motion modeling module into the frozen text-to-image model and train it on video clips to distill reasonable motion priors. Once trained, by simply injecting this motion modeling module, all personalized versions derived from the same base T2I readily become text-driven models that produce diverse and personalized animated images. We conduct our evaluation on several public representative personalized text-to-image models across anime pictures and realistic photographs, and demonstrate that our proposed framework helps these models generate temporally smooth animation clips while preserving the domain and diversity of their outputs.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/230302421-a9107d03-92d3-44b1-91b4-fde4ad2725d4.png">
</div>

## Pretrained models

We use Stable Diffusion's weights provided by HuggingFace Diffusers. You do not have to download the weights manually. If you use Diffusers wrapper, the weights will be downloaded automatically.

This model has several weights including vae, unet and clip. You should download the weights from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and change the 'pretrained_model_path' in config to the weights dir.


## Quick Start

Running the following codes, you can get a text-generated image.

```python
import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

import os
import torch
from pathlib import Path
import datetime
from mmagic.models.editors.animatediff import save_videos_grid



register_all_modules()

cfg = Config.fromfile('configs/animatediff/animatediff_ToonYou.py')
animatediff = MODELS.build(cfg.model).cuda()
# breakpoint()
prompts = [
    "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress",

    "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes,",

    "best quality, masterpiece, 1boy, formal, abstract, looking at viewer, masculine, marble pattern",

    "best quality, masterpiece, 1girl, cloudy sky, dandelion, contrapposto, alternate hairstyle,"
]

negative_prompts = [
    "",
    "badhandv4,easynegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3, bad-artist, bad_prompt_version2-neg, teeth",
    "",
    "",
]

sample_idx = 0
random_seeds = cfg.randomness['seed']
random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
samples = []
time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
savedir = f"samples/{Path(cfg.model['dream_booth_lora_cfg']['type']).stem}-{time_str}"
os.makedirs(savedir)
for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, negative_prompts, random_seeds)):
    output_dict = animatediff.infer(prompt,negative_prompt=n_prompt, video_length=16, height=256, width=256, seed=random_seed,num_inference_steps=cfg.model['dream_booth_lora_cfg']['steps'])
    sample = output_dict['samples']
    prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
    save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
    print(f"save to {savedir}/sample/{prompt}.gif")
    samples.append(sample)
    sample_idx += 1

samples = torch.concat(samples)
save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)


```

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/28132635/230297033-4f5c32df-365c-4cf4-8e4f-1b76a4cbb0b7.png" width="400"/>
  <br/>
  <b>'control_0.png'</b>
</div></td>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/28132635/230298159-a25695f8-fee4-40b2-aec0-01566ab25a97.png" width="400"/>
  <br/>
  <b>'sample_0.png'</b>
</div></td>
    <td>
</thead>
</table>

### Using MMInferencer

Ongoing...


## Citation

```bibtex
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Wang, Yaohui and Qiao, Yu and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2307.04725},
  year={2023}
}
```
