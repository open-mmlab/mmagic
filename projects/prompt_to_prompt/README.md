# Inversions & Editing (DDIM Inversion & Null-Text Inversion & Prompt-to-Prompt Editing)

```
Author: @FerryHuang

This is an implementation of the papers:
```

> [PROMPT-TO-PROMPT IMAGE EDITING
> WITH CROSS-ATTENTION CONTROL](https://prompt-to-prompt.github.io/ptp_files/Prompt-to-Prompt_preprint.pdf)

> [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://arxiv.org/pdf/2211.09794.pdf)

> **Task**: Text2Image, diffusion, inversion, editing

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Diffusion's inversion basically means you put an image (with or without a prompt) into a method and it will return a latent code which can be later turned back to a image with high simmilarity as the original one. Of course we want this latent code for an editing purpose, that's also why we always implement inversion methods together with the editing methods.

This project contains **Two inversion methods** and **One editing method**.

## From right to left: origin image, DDIM inversion, Null-text inversion

<center class="half">
    <img src="https://github.com/FerryHuang/mmagic/assets/71176040/34d8a467-5378-41fb-83c6-b23c9dee8f0a" width="200"/><img src="https://github.com/FerryHuang/mmagic/assets/71176040/3d3814b4-7fb5-4232-a56f-fd7fef0ba28e" width="200"/><img src="https://github.com/FerryHuang/mmagic/assets/71176040/43008ed4-a5a3-4f81-ba9f-95d9e79e6a08" width="200"/>
</center>

## Prompt-to-prompt Editing

<div align="center">
  <b>cat -> dog</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/f5d3fc0c-aa7b-4525-9364-365b254d51ca" width="500"/>
</div>

<div align="center">
  <b>spider man -> iron man(attention replace)</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/074adbc6-bd48-4c82-99aa-f322cf937f5a" width="500"/>
</div>

<div align="center">
  <b>Effel tower -> Effel tower at night (attention refine)</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/f815dab3-b20c-4936-90e3-a060d3717e22" width="500"/>
</div>

<div align="center">
  <b>blossom sakura tree -> blossom(-3) sakura tree (attention reweight)</b>
  <br/>
  <img src="https://github.com/FerryHuang/mmagic/assets/71176040/5ef770b9-4f28-4ae7-84b0-6c15ea7450e9" width="500"/>
</div>

## Quick Start

A walkthrough of the project is provided [here](visualize.ipynb)

or you can just run the following scripts to get the results:

```python
# load the mmagic SD1.5
from mmengine import MODELS, Config
from mmengine.registry import init_default_scope

init_default_scope('mmagic')

config = 'configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
config = Config.fromfile(config).copy()

StableDiffuser = MODELS.build(config.model)
StableDiffuser = StableDiffuser.to('cuda')
```

```python
# inversion
image_path = 'projects/prompt_to_prompt/assets/gnochi_mirror.jpeg'
prompt = "a cat sitting next to a mirror"
image_tensor = ptp_utils.load_512(image_path).to('cuda')

from inversions.null_text_inversion import NullTextInversion
from models.ptp import EmptyControl
from models import ptp_utils

null_inverter = NullTextInversion(StableDiffuser)
null_inverter.init_prompt(prompt)
ddim_latents = null_inverter.ddim_inversion(image_tensor)
x_t = ddim_latents[-1]
uncond_embeddings = null_inverter.null_optimization(ddim_latents, num_inner_steps=10, epsilon=1e-5)
null_text_rec, _ = ptp_utils.text2image_ldm_stable(StableDiffuser, [prompt], EmptyControl(), latent=x_t, uncond_embeddings=uncond_embeddings)
ptp_utils.view_images(null_text_rec)
```

```python
# prompt-to-prompt editing
prompts = ["A cartoon of spiderman",
           "A cartoon of ironman"]
import torch
from models.ptp import LocalBlend, AttentionReplace
from models.ptp_utils import text2image_ldm_stable
g = torch.Generator().manual_seed(2023616)
lb = LocalBlend(prompts, ("spiderman", "ironman"), model=StableDiffuser)
controller = AttentionReplace(prompts, 50,
                              cross_replace_steps={"default_": 1., "ironman": .2},
                              self_replace_steps=0.4,
                              local_blend=lb, model=StableDiffuser)
images, x_t = text2image_ldm_stable(StableDiffuser, prompts, controller, latent=None,
                                    num_inference_steps=50, guidance_scale=7.5, uncond_embeddings=None, generator=g)
```

## Citation

```bibtex
@article{hertz2022prompt,
  title = {Prompt-to-Prompt Image Editing with Cross Attention Control},
  author = {Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2208.01626},
  year = {2022},
}
@article{mokady2022null,
  title={Null-text Inversion for Editing Real Images using Guided Diffusion Models},
  author={Mokady, Ron and Hertz, Amir and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2211.09794},
  year={2022}
}
```
