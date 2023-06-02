# GLIDE (Arxiv'2021)

> [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)

> **Task**: Text2Image, diffusion

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Diffusion models have recently been shown to generate high-quality synthetic images, especially when paired with a guidance technique to trade off diversity for fidelity. We explore diffusion models for the problem of text-conditional image synthesis and compare two different guidance strategies: CLIP guidance and classifier-free guidance. We find that the latter is preferred by human evaluators for both photorealism and caption similarity, and often produces photorealistic samples. Samples from a 3.5 billion parameter text-conditional diffusion model using classifierfree guidance are favored by human evaluators to those from DALL-E, even when the latter uses expensive CLIP reranking. Additionally, we find that our models can be fine-tuned to perform image inpainting, enabling powerful text-driven image editing. We train a smaller model on a filtered dataset and release the code and weights at https://github.com/openai/glide-text2im.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/22982797/209770463-31f3083d-b939-4ed6-b504-6a5baf7365b5.png" width="400"/>
</div >

## Results and models

<div align="center">
  <b>an oil painting of a corgi</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/210042533-1df54b2d-d8a8-42b1-974c-06861e3e6ef6.png" width="400"/>
</div>

<div align="center">
  <b>an cartoon painting of a cat</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/210042530-ada31a01-7c9d-452b-bc72-56ae0182ef2f.png" width="400"/>
</div>

**Laion**

| Method | Resolution       | Config                                                                      | Weights                                                                                    |
| ------ | ---------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Glide  | 64x64            | [config](projects/glide/configs/glide_ddim-classifier-free_laion-64x64.py)  | [model](https://download.openmmlab.com/mmediting/glide/glide_laion-64x64-02afff47.pth)     |
| Glide  | 64x64 -> 256x256 | [config](projects/glide/configs/glide_ddim-classifier-free_laion-64-256.py) | [model](https://download.openxlab.org.cn/models/mmediting/GLIDE/weight/glide_laion-64-256) |

## Quick Start

You can run glide as follows:

```python
import torch
from mmagic.apis import init_model
from mmengine.registry import init_default_scope
from projects.glide.models import *

init_default_scope('mmagic')

config = 'projects/glide/configs/glide_ddim-classifier-free_laion-64x64.py'
ckpt = 'https://download.openmmlab.com/mmagic/glide/glide_laion-64x64-02afff47.pth'
model = init_model(config, ckpt).cuda().eval()
prompt = "an oil painting of a corgi"

with torch.no_grad():
    samples = model.infer(init_image=None,
                prompt=prompt,
                batch_size=16,
                guidance_scale=3.,
                num_inference_steps=100,
                labels=None,
                classifier_scale=0.0,
                show_progress=True)['samples']
```

You can synthesis images with 256x256 resolution:

```python
import torch
from torchvision.utils import save_image
from mmagic.apis import init_model
from mmengine.registry import init_default_scope
from projects.glide.models import *

init_default_scope('mmagic')

config = 'projects/glide/configs/glide_ddim-classifier-free_laion-64-256.py'
ckpt = 'https://download.openxlab.org.cn/models/mmediting/GLIDE/weight/glide_laion-64-256'
model = init_model(config, ckpt).cuda().eval()
prompt = "an oil painting of a corgi"

with torch.no_grad():
    samples = model.infer(init_image=None,
                prompt=prompt,
                batch_size=16,
                guidance_scale=3.,
                num_inference_steps=100,
                labels=None,
                classifier_scale=0.0,
                show_progress=True)['samples']
save_image(samples, "corgi.png", nrow=4, normalize=True, value_range=(-1, 1))
```

## Citation

```bibtex
@article{2021GLIDE,
  title={GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models},
  author={ Nichol, A.  and  Dhariwal, P.  and  Ramesh, A.  and  Shyam, P.  and  Mishkin, P.  and  Mcgrew, B.  and  Sutskever, I.  and  Chen, M. },
  year={2021},
}
```
