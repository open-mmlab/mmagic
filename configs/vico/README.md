# ViCo (2023)

> [ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation](https://arxiv.org/abs/2306.00971)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Personalized text-to-image generation using diffusion models has recently been proposed and attracted lots of attention. Given a handful of images containing a novel concept (e.g., a unique toy), we aim to tune the generative model to capture fine visual details of the novel concept and generate photorealistic images following a text condition. We present a plug-in method, named ViCo, for fast and lightweight personalized generation. Specifically, we propose an image attention module to condition the diffusion process on the patch-wise visual semantics. We introduce an attention-based object mask that comes almost at no cost from the attention module. In addition, we design a simple regularization based on the intrinsic properties of text-image attention maps to alleviate the common overfitting degradation. Unlike many existing models, our method does not finetune any parameters of the original diffusion model. This allows more flexible and transferable model deployment. With only light parameter training (~6% of the diffusion U-Net), our method achieves comparable or even better performance than all state-of-the-art models both qualitatively and quantitatively.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/haoosz/ViCo/assets/71176040/0ee95a57-fecf-4bba-bc64-eda46e5cc6d1">
</div>

## Configs

|       Model       | Dataset | Download |
| :---------------: | :-----: | :------: |
| [ViCo](./vico.py) |    -    |    -     |

## Quick Start

1. Download concept data and imagenet_templates_small.txt from [here](https://openxlab.org.cn/datasets/ferry/ViCo/tree/main).
   and save to `data/vico/`

The file structure will be like this:

```text
data
└── vico
    └──batman
       ├── 1.jpg
       ├── 2.jpg
       ├── 3.jpg
       └── 4.jpg
    └──clock
       ├── 1.jpg
       ├── 2.jpg
       ├── 3.jpg
       └── 4.jpg
    ...
    └──imagenet_templates_small.txt
```

2. Customize your config

```
# Only need to care about these

# which concept you want to customize
concept_dir = 'dog7'

# the new token to denote the concept
placeholder: str = 'S*'

# better to be the superclass of concept
initialize_token: str = 'dog'
```

3. Start training with the following command:

```bash
# 4 GPUS
bash tools/dist_train.sh configs/vico/vico.py 4
# 1 GPU
python tools/train.py configs/vico/vico.py
```

4. Use the [pretrained checkpoins](https://openxlab.org.cn/models/detail/ferry/ViCo) to inference

```python
import torch
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

# say you have downloaded the pretrained weights
cfg = Config.fromfile('configs/vico/dog.py')
state_dict = torch.load("./dog.pth")
vico = MODELS.build(cfg.model)
vico.load_state_dict(state_dict, strict=False)
vico = vico.cuda()

prompt = ["A photo of S*", "A photo of S* on the beach"]
reference = "data/vico/dog7/01.jpg"
image_ref = Image.open(reference)
with torch.no_grad():
    output = vico.infer(prompt=prompt, image_reference=image_ref, seed=123, num_images_per_prompt=2)['samples'][0]
output.save("infer.png")
```

5. (Optional) If you want to use the weight trained by the
   commands at step3, here are codes to extract the trained parameters, then you can infer with it like step4

```python
import torch
def extract_vico_parameters(state_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if 'image_cross_attention' in k or 'trainable_embeddings' in k:
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
    return new_state_dict

checkpoint = torch.load("work_dirs/vico/iter_400.pth")
new_checkpoint = extract_vico_parameters(checkpoint['state_dict'])
torch.save(new_checkpoint, "work_dirs/vico/dog.pth")
```

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://github.com/open-mmlab/mmagic/assets/71176040/58a6953c-053a-40ea-8826-eee428c992b5" width="800"/>
  <br/>
  <b>'vico'</b>
</thead>
</table>

## Comments

Our codebase for the stable diffusion models builds heavily on [diffusers codebase](https://github.com/huggingface/diffusers) and the model weights are from [stable-diffusion-1.5](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py).

Thanks for the efforts of the community!

## Citation

```bibtex
@inproceedings{Hao2023ViCo,
  title={ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation},
  author={Shaozhe Hao and Kai Han and Shihao Zhao and Kwan-Yee K. Wong},
  year={2023}
}
```
