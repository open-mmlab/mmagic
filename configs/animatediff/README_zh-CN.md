# AnimateDiff (2023)

> [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)

> **任务**: 视频生成, 扩散模型

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

年来，AIGC 宛如 AI 海洋中最不可或缺的波涛，逐渐凝成滔天的巨浪，突破壁垒、扑向海岸，并酝酿着下一波潮水高涨。以 Stable Diffusion 这股翻腾最为汹涌的波涛为代表的文生图模型飞速发展，使得更多非专业用户也能通过简单的文字提示生成高质量的图片内容。然而，文生图模型的训练成本往往十分高昂，为减轻微调模型的代价，相应的模型定制化方法如 DreamBooth, LoRA 应运而生，使得用户在开源权重的基础上，用少量数据和消费级显卡即可实现模型个性化和特定风格下的图像生成质量的提升。这极大推动了 HuggingFace, CivitAI 等开源模型社区的发展，众多艺术家和爱好者在其中贡献了许多高质量的微调模型。不觉间，平静的海洋洪水滔天，海滩上留下数不清的色彩斑斓的鹅卵石，便是爱好者们精心调制的 AI 画作。

与动画相比，静态图像的表达能力是有限的。随着越来越多效果惊艳的微调模型的出现和视频生成技术的发展，人们期待着能够赋予这些定制化模型生成动画的能力。在最新开源的 AnimateDiff 中，作者提出了一种将任何定制化文生图模型拓展用于动画生成的框架，可以在保持原有定制化模型画面质量的基础上，生成相应的动画片段。为色彩斑斓的鹅卵石，增添一些动态的光泽。

<!-- [IMAGE] -->

![512](https://github.com/ElliotQi/mmagic/assets/46469021/54d92aca-dfa9-4eeb-ba38-3f6c981e5399)

## 模型与结果

我们使用HuggingFace提供的Stable Diffusion权重。如果您使用Diffusers wrapper，您不必手动下载权重，其将自动下载。

<!-- SKIP THIS TABLE -->

|                              模型                              |                                                下载                                                 |
| :------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
|              [ToonYou](./animatediff_ToonYou.py)               |                       [model](https://civitai.com/api/download/models/78775)                        |
|               [Lyriel](./animatediff_Lyriel.py)                |                       [model](https://civitai.com/api/download/models/72396)                        |
|          [RcnzCartoon](./animatediff_RcnzCartoon.py)           |                       [model](https://civitai.com/api/download/models/71009)                        |
|             [MajicMix](./animatediff_MajicMix.py)              |                       [model](https://civitai.com/api/download/models/79068)                        |
|      [RealisticVision](./animatediff_RealisticVision.py)       |                       [model](https://civitai.com/api/download/models/29460)                        |
|   [RealisticVision_v2](./animatediff_RealisticVision_v2.py)    |          [model](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt)          |
|   [MotionModel_v1-5_v2](./animatediff_RealisticVision_v2.py)   |      [model](https://download.openxlab.org.cn/models/ElliotQi/AnimateDiff/weight/mm_sd_v15_v2)      |
| [MotionModel_v1-5_2Mval](./animatediff_RealisticVision_v1.py)  | [model](https://download.openxlab.org.cn/models/ElliotQi/AnimateDiff/weight/mm_fromscratch_2.5Mval) |
| [MotionModel_v1-5_10Mval](./animatediff_RealisticVision_v1.py) |                                               WebVid                                                |

最新模型更新请在[浦源_AnimateDiff](https://openxlab.org.cn/models/detail/ElliotQi/AnimateDiff)中查看

## 快速开始

运行以下代码，你可以使用AnimateDiff通过文本生成视频。

#### 小建议

强烈推荐安装 [xformers](https://github.com/facebookresearch/xformers)，512\*512分辨率可以节省约20G显存。

1. 下载 [ToonYou](https://civitai.com/api/download/models/78775) 和 MotionModule 权重

```bash
#!/bin/bash

mkdir models && cd models
mkdir Motion_Module && mkdir DreamBooth_LoRA
gdown 1RqkQuGPaCO5sGZ6V6KZ-jUWmsRu48Kdq -O Motion_Module/
gdown 1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu -O models/Motion_Module/
wget https://civitai.com/api/download/models/78775 -P DreamBooth_LoRA/ --content-disposition --no-check-certificate
```

2. 修改 `configs/animatediff/animatediff_ToonYou.py` 配置文件中的权重路径

```python
    models_path = {Your Checkpoints Path}
    motion_module_cfg=dict(
        path={Your MotionModule Path}
    ),
    dream_booth_lora_cfg=dict(
        type='ToonYou',
        path={Your Dreambooth_Lora Path},
        steps=25,
        guidance_scale=7.5)
```

3. 享受AnimateDiff视频生成吧！

```python
from mmengine import Config
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

### 其他配置文件的Prompts

- Lyriel

```yaml
  prompt:
    - "dark shot, epic realistic, portrait of halo, sunglasses, blue eyes, tartan scarf, white hair by atey ghailan, by greg rutkowski, by greg tocchini, by james gilleard, by joe fenton, by kaethe butcher, gradient yellow, black, brown and magenta color scheme, grunge aesthetic!!! graffiti tag wall background, art by greg rutkowski and artgerm, soft cinematic light, adobe lightroom, photolab, hdr, intricate, highly detailed, depth of field, faded, neutral colors, hdr, muted colors, hyperdetailed, artstation, cinematic, warm lights, dramatic light, intricate details, complex background, rutkowski, teal and orange"
    - "A forbidden castle high up in the mountains, pixel art, intricate details2, hdr, intricate details, hyperdetailed5, natural skin texture, hyperrealism, soft light, sharp, game art, key visual, surreal"
    - "dark theme, medieval portrait of a man sharp features, grim, cold stare, dark colors, Volumetric lighting, baroque oil painting by Greg Rutkowski, Artgerm, WLOP, Alphonse Mucha dynamic lighting hyperdetailed intricately detailed, hdr, muted colors, complex background, hyperrealism, hyperdetailed, amandine van ray"
    - "As I have gone alone in there and with my treasures bold, I can keep my secret where and hint of riches new and old. Begin it where warm waters halt and take it in a canyon down, not far but too far to walk, put in below the home of brown."

  n_prompt:
    - "3d, cartoon, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, young, loli, elf, 3d, illustration"
    - "3d, cartoon, anime, sketches, worst quality, low quality, normal quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, bad anatomy, girl, loli, young, large breasts, red eyes, muscular"
    - "dof, grayscale, black and white, bw, 3d, cartoon, anime, sketches, worst quality, low quality, normal quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, bad anatomy, girl, loli, young, large breasts, red eyes, muscular,badhandsv5-neg, By bad artist -neg 1, monochrome"
    - "holding an item, cowboy, hat, cartoon, 3d, disfigured, bad art, deformed,extra limbs,close up,b&w, weird colors, blurry, duplicate, morbid, mutilated, [out of frame], extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"
```

- RcnzCartoon

```yaml
prompt:
    - "Jane Eyre with headphones, natural skin texture,4mm,k textures, soft cinematic light, adobe lightroom, photolab, hdr, intricate, elegant, highly detailed, sharp focus, cinematic look, soothing tones, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded"
    - "close up Portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal [rust], elegant, sharp focus, photo by greg rutkowski, soft lighting, vibrant colors, masterpiece, streets, detailed face"
    - "absurdres, photorealistic, masterpiece, a 30 year old man with gold framed, aviator reading glasses and a black hooded jacket and a beard, professional photo, a character portrait, altermodern, detailed eyes, detailed lips, detailed face, grey eyes"
    - "a golden labrador, warm vibrant colours, natural lighting, dappled lighting, diffused lighting, absurdres, highres,k, uhd, hdr, rtx, unreal, octane render, RAW photo, photorealistic, global illumination, subsurface scattering"

  n_prompt:
    - "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
    - "nude, cross eyed, tongue, open mouth, inside, 3d, cartoon, anime, sketches, worst quality, low quality, normal quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, bad anatomy, red eyes, muscular"
    - "easynegative, cartoon, anime, sketches, necklace, earrings worst quality, low quality, normal quality, bad anatomy, bad hands, shiny skin, error, missing fingers, extra digit, fewer digits, jpeg artifacts, signature, watermark, username, blurry, chubby, anorectic, bad eyes, old, wrinkled skin, red skin, photograph By bad artist -neg, big eyes, muscular face,"
    - "beard, EasyNegative, lowres, chromatic aberration, depth of field, motion blur, blurry, bokeh, bad quality, worst quality, multiple arms, badhand"

```

- MajicMix

```yaml
prompt:
    - "1girl, offshoulder, light smile, shiny skin best quality, masterpiece, photorealistic"
    - "best quality, masterpiece, photorealistic, 1boy, 50 years old beard, dramatic lighting"
    - "best quality, masterpiece, photorealistic, 1girl, light smile, shirt with collars, waist up, dramatic lighting, from below"
    - "male, man, beard, bodybuilder, skinhead,cold face, tough guy, cowboyshot, tattoo, french windows, luxury hotel masterpiece, best quality, photorealistic"

  n_prompt:
    - "ng_deepnegative_v1_75t, badhandv4, worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, watermark, moles"
    - "nsfw, ng_deepnegative_v1_75t,badhandv4, worst quality, low quality, normal quality, lowres,watermark, monochrome"
    - "nsfw, ng_deepnegative_v1_75t,badhandv4, worst quality, low quality, normal quality, lowres,watermark, monochrome"
    - "nude, nsfw, ng_deepnegative_v1_75t, badhandv4, worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, monochrome, grayscale watermark, moles, people"
```

- Realistic & Realistic_v2 (两者使用相同prompts和不同的随机种子)

```yaml
  prompt:
    - "b&w photo of 42 y.o man in black clothes, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    - "close up photo of a rabbit, forest, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot"
    - "photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    - "night, b&w photo of old house, post apocalypse, forest, storm weather, wind, rocks, 8k uhd, dslr, soft lighting, high quality, film grain"

  n_prompt:
    - "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    - "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    - "blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
    - "blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, art, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"

```

4. 开始训练运动模块脚本:

```bash
# 4 GPUS
bash tools/dist_train.sh configs/animatediff/animatediff.py 4
# 1 GPU
python tools/train.py configs/animatediff/animatediff.py

```

## 引用

```bibtex
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Wang, Yaohui and Qiao, Yu and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2307.04725},
  year={2023}
}
```
