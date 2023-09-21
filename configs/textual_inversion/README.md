# Textual Inversion (2022)

> [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Text-to-image models offer unprecedented freedom to guide creation through natural language. Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes. In other words, we ask: how can we use language-guided models to turn our cat into a painting, or imagine a new product based on our favorite toy? Here we present a simple approach that allows such creative freedom. Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model. These "words" can be composed into natural language sentences, guiding personalized creation in an intuitive way. Notably, we find evidence that a single word embedding is sufficient for capturing unique and varied concepts. We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmagic/assets/28132635/b2dac6f1-5151-4199-bcc2-71b5b1523a16">
</div>

## Configs

|                    Model                    | Dataset | Download |
| :-----------------------------------------: | :-----: | :------: |
| [Textual Inversion](./textual_inversion.py) |    -    |    -     |

## Quick Start

1. Download [data](https://drive.google.com/drive/folders/1fmJMs25nxS_rSNqS5hTcRdLem_YQXbq5) and [template](https://openxlab.org.cn/datasets/ferry/ViCo/tree/main)(two txt files)
   and save to `data`

The file structure will be like this:

```text
data
└── cat_toy
    ├── 1.jpeg
    ├── 2.jpeg
    ├── 3.jpeg
    ├── 3.jpeg
    ├── 4.jpeg
    ├── 6.jpeg
    └── 7.jpeg
└── imagenet_templates_small.txt
└── imagenet_style_templates_small.txt
```

2. Start training with the following command:

```bash
bash tools/dist_train.sh configs/textual_inversion/textual_inversion.py 1
```

<div align="center">
  <img src="https://github.com/open-mmlab/mmagic/assets/28132635/635a336c-fd6c-4c6f-b2c1-c1621420b9b9" width="400"/>
  <br/>
</div>

3. Inference with trained textual embedding:

```python
import torch
from mmengine import Config

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


def process_state_dict(state_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v

    return new_state_dict


cfg = Config.fromfile('configs/textual_inversion/textual_inversion.py')
checkpoint = torch.load('work_dirs/textual_inversion/iter_3000.pth')
state_dict = process_state_dict(checkpoint['state_dict'])
model = MODELS.build(cfg.model)
model.load_state_dict(state_dict)

model = model.cuda()
with torch.no_grad():
    sample = model.infer('a <cat-toy> bag')['samples'][0]

sample.save('cat-toy-bag.png')
```

## Comments

Our codebase for the stable diffusion models builds heavily on [diffusers codebase](https://github.com/huggingface/diffusers) and the model weights are from [stable-diffusion-1.5](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py).

Thanks for the efforts of the community!

## Citation

```bibtex
@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}

```
