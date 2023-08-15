# AnimateDiff (2023)

> [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)

> **任务**: 视频生成, 扩散模型

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

随着文本到图像模型（例如，稳定扩散）和相应的个性化技术（例如，LoRA和DreamBooth）的进步，每个人都有可能以较低的成本将他们的想象力展现在高质量的图像中。在这个项目中，我们提出了一个有效的框架（AnimateDiff），可以一次性为大多数现有的个性化文本到图像模型制作动画，从而节省了特定模型调整的工作量。

拟议框架的核心是将新初始化的运动建模模块附加到基于冻结的文本到图像模型中，并在此后在视频剪辑上对其进行训练，以便事先提炼出合理的运动。一旦经过培训，只需注入此运动建模模块，所有来自同一基础的个性化版本都很容易成为文本驱动模型，可以生成多样化和个性化的动画图像。

<!-- [IMAGE] -->

TODO... Add Image

## 模型与结果

我们使用HuggingFace提供的Stable Diffusion权重。如果您使用Diffusers wrapper，您不必手动下载权重，其将自动下载。

<!-- SKIP THIS TABLE -->

|  模型   |     下载     |
| :-----: | :----------: |
| ToonYou | Coming soon! |

## 待办列表

- [x] 整体pipeline完成
- [x] 支持xformer显存优化，目前可以在13G显存下输出16帧512\*512视频
- \[\] 优化512\*512视频质量
- \[\] 支持以图生成视频
- \[\] 完成Gradio部署
- \[\] 训练SD XL基础上的Motion Module
- \[\] 支持更快的采样器(plms，dpm-solver等)

我们很欢迎社区用户支持这些项目和任何其他有趣的工作!

## Quick Start

运行以下代码，你可以使用文本生成图像。

Running the following codes, you can get a text-generated image.

1. 下载 [ToonYou](https://civitai.com/api/download/models/78775) 和 MotionModule 权重

```bash
#!/bin/bash

gdown 1RqkQuGPaCO5sGZ6V6KZ-jUWmsRu48Kdq -O models/Motion_Module/
gdown 1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu -O models/Motion_Module/
wget https://civitai.com/api/download/models/78775 -P models/DreamBooth_LoRA/ --content-disposition --no-check-certificate
```

2. 修改 `configs/animatediff/animatediff_ToonYou.py` 配置文件中的权重路径

```python

    motion_module_cfg=dict(
        path={Your MotionModule path}
    ),
    dream_booth_lora_cfg=dict(
        type='ToonYou',
        path={Your Dreambooth_Lora path},
        steps=25,
        guidance_scale=7.5))
```

3. 享受AnimateDiff视频生成吧！

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

## Citation

```bibtex
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Wang, Yaohui and Qiao, Yu and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2307.04725},
  year={2023}
}
```
