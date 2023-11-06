# AnimateDiff (2023)

> [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)

> **Task**: Text2Video

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

With the advance of text-to-image models (e.g., Stable Diffusion) and corresponding personalization techniques such as DreamBooth and LoRA, everyone can manifest their imagination into high-quality images at an affordable cost. Subsequently, there is a great demand for image animation techniques to further combine generated static images with motion dynamics. In this report, we propose a practical framework to animate most of the existing personalized text-to-image models once and for all, saving efforts in model-specific tuning. At the core of the proposed framework is to insert a newly initialized motion modeling module into the frozen text-to-image model and train it on video clips to distill reasonable motion priors. Once trained, by simply injecting this motion modeling module, all personalized versions derived from the same base T2I readily become text-driven models that produce diverse and personalized animated images. We conduct our evaluation on several public representative personalized text-to-image models across anime pictures and realistic photographs, and demonstrate that our proposed framework helps these models generate temporally smooth animation clips while preserving the domain and diversity of their outputs.

<!-- [IMAGE] -->

![512](https://github.com/ElliotQi/mmagic/assets/46469021/54d92aca-dfa9-4eeb-ba38-3f6c981e5399)

## Pretrained models

We use Stable Diffusion's weights provided by HuggingFace Diffusers. You do not have to download the weights manually. If you use Diffusers wrapper, the weights will be downloaded automatically.

This model has several weights including vae, unet and clip. You should download the weights from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and change the 'pretrained_model_path' in config to the weights dir.

|                             Model                              | Dataset |                                                      Download                                                       |
| :------------------------------------------------------------: | :-----: | :-----------------------------------------------------------------------------------------------------------------: |
|              [ToonYou](./animatediff_ToonYou.py)               |    -    |        [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/toonyou_beta3.safetensors)        |
|               [Lyriel](./animatediff_Lyriel.py)                |    -    |         [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/lyriel_v16.safetensors)          |
|          [RcnzCartoon](./animatediff_RcnzCartoon.py)           |    -    |      [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/rcnzCartoon3d_v10.safetensors)      |
|             [MajicMix](./animatediff_MajicMix.py)              |    -    | [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/majicmixRealistic_v5Preview.safetensors) |
|      [RealisticVision](./animatediff_RealisticVision.py)       |    -    | [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/realisticVisionV51_v20Novae.safetensors) |
|   [MotionModel_v1-5_v2](./animatediff_RealisticVision_v2.py)   | WebVid  |            [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/mm_sd_v15_v2.ckpt)            |
| [MotionModel_v1-5_2Mval](./animatediff_RealisticVision_v1.py)  | WebVid  |       [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/mm_fromscratch_2.5Mval.ckpt)       |
| [MotionModel_v1-5_10Mval](./animatediff_RealisticVision_v1.py) | WebVid  |       [model](https://download.openxlab.org.cn/models/Masbfca/AnimateDiff/weight/mm_fromscratch_10Mval.ckpt)        |

Latest models could be looked up on [OpenXLab_AnimateDiff](https://openxlab.org.cn/models/detail/ElliotQi/AnimateDiff).

## Quick Start

Running the following codes, you can get a text-generated image.

### Reccomendation

It's highly recommended to install [xformers](https://github.com/facebookresearch/xformers). It would save about 20G memory for 512\*512 resolution generation.

### Steps

1. Download [ToonYou](https://civitai.com/api/download/models/78775) and MotionModule checkpoint

```bash
#!/bin/bash

mkdir models && cd models
mkdir Motion_Module && mkdir DreamBooth_LoRA
gdown 1RqkQuGPaCO5sGZ6V6KZ-jUWmsRu48Kdq -O Motion_Module/
gdown 1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu -O models/Motion_Module/
wget https://civitai.com/api/download/models/78775 -P DreamBooth_LoRA/ --content-disposition --no-check-certificate
```

2. Modify the config file in `configs/animatediff/animatediff_ToonYou.py`

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

3. Enjoy Text2Video world

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

### Prompts for other config

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

- Realistic & Realistic_v2 (same prompts with different random seed, find more details in their config files)

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

4. Start training motion module with the following command:

```bash
# 4 GPUS
bash tools/dist_train.sh configs/animatediff/animatediff.py 4
# 1 GPU
python tools/train.py configs/animatediff/animatediff.py

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
