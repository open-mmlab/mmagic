# Animated Drawings (SIGGRAPH'2023)

> [A Method for Animating Children's Drawings of The Human Figure](https://dl.acm.org/doi/10.1145/3592788)

> **Task**: Drawing

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Children’s drawings have a wonderful inventiveness, creativity, and variety to them. We present a system that automatically animates children’s drawings of the human figure, is robust to the variance inherent in these depictions, and is simple and straightforward enough for anyone to use. We demonstrate the value and broad appeal of our approach by building and releasing the Animated Drawings Demo, a freely available public website that has been used by millions of people around the world. We present a set of experiments exploring the amount of training data needed for fine-tuning, as well as a perceptual study demonstrating the appeal of a novel twisted perspective retargeting technique. Finally, we introduce the Amateur Drawings Dataset, a first-of-its-kind annotated dataset, collected via the public demo, containing over 178,000 amateur drawings and corresponding user-accepted character bounding boxes, segmentation masks, and joint location annotations.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/6675724/219223438-2c93f9cb-d4b5-45e9-a433-149ed76affa6.gif" width="400"/>
</div >

## Results





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

## Citation

```bibtex
@article{2021GLIDE,
  title={GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models},
  author={ Nichol, A.  and  Dhariwal, P.  and  Ramesh, A.  and  Shyam, P.  and  Mishkin, P.  and  Mcgrew, B.  and  Sutskever, I.  and  Chen, M. },
  year={2021},
}
```
