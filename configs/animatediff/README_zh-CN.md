# AnimateDiff (2023)

> [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)

> **任务**: 视频生成, 扩散模型

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

年来，AIGC 宛如 AI 海洋中最不可或缺的波涛，逐渐凝成滔天的巨浪，突破壁垒、扑向海岸，并酝酿着下一波潮水高涨。以 Stable Diffusion 这股翻腾最为汹涌的波涛为代表的文生图模型飞速发展，使得更多非专业用户也能通过简单的文字提示生成高质量的图片内容。然而，文生图模型的训练成本往往十分高昂，为减轻微调模型的代价，相应的模型定制化方法如 DreamBooth, LoRA 应运而生，使得用户在开源权重的基础上，用少量数据和消费级显卡即可实现模型个性化和特定风格下的图像生成质量的提升。这极大推动了 HuggingFace, CivitAI 等开源模型社区的发展，众多艺术家和爱好者在其中贡献了许多高质量的微调模型。不觉间，平静的海洋洪水滔天，海滩上留下数不清的色彩斑斓的鹅卵石，便是爱好者们精心调制的 AI 画作。

与动画相比，静态图像的表达能力是有限的。随着越来越多效果惊艳的微调模型的出现和视频生成技术的发展，人们期待着能够赋予这些定制化模型生成动画的能力。在最新开源的 AnimateDiff 中，作者提出了一种将任何定制化文生图模型拓展用于动画生成的框架，可以在保持原有定制化模型画面质量的基础上，生成相应的动画片段。为色彩斑斓的鹅卵石，增添一些动态的光泽。

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

运行以下代码，你可以使用AnimateDiff通过文本生成视频。

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
