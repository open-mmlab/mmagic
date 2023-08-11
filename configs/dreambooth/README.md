# DreamBooth (2022)

> [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for "personalization" of text-to-image diffusion models. Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can be used to synthesize novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while preserving the subject's key features. We also provide a new dataset and evaluation protocol for this new task of subject-driven generation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/232406756-04dc1fbe-abde-4bb5-950c-ad3e125d5252.png">
</div>

## Configs

|                                     Model                                      | Dataset | Download |
| :----------------------------------------------------------------------------: | :-----: | :------: |
|                         [DreamBooth](./dreambooth.py)                          |    -    |    -     |
|  [DreamBooth (Finetune Text Encoder)](./dreambooth-finetune_text_encoder.py)   |    -    |    -     |
|      [DreamBooth with Prior-Preservation Loss](./dreambooth-prior_pre.py)      |    -    |    -     |
|                    [DreamBooth LoRA](./dreambooth-lora.py)                     |    -    |    -     |
| [DreamBooth LoRA with Prior-Preservation Loss](./dreambooth-lora-prior_pre.py) |    -    |    -     |

## Quick Start

1. Download [data](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ) and save to `data/dreambooth/`

The file structure will be like this:

```text
data
└── dreambooth
    └──imgs
       ├── alvan-nee-Id1DBHv4fbg-unsplash.jpeg
       ├── alvan-nee-bQaAJCbNq3g-unsplash.jpeg
       ├── alvan-nee-brFsZ7qszSY-unsplash.jpeg
       └── alvan-nee-eoqnr8ikwFE-unsplash.jpeg
```

2. Start training with the following command:

```bash
bash tools/dist_train.sh configs/dreambooth/dreambooth.py 1
# or
bash tools/dist_train.sh configs/dreambooth/dreambooth-lora.py 1
```

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/28132635/232682088-26424e69-f697-49bc-a706-d03245ff25b1.png" width="400"/>
  <br/>
  <b>'dreambooth'</b>
</div></td>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/28132635/232682057-fbc99047-e2d0-433e-bbc5-4f2d4ec18191.png" width="400"/>
  <br/>
  <b>'dreambooth-lora'</b>
</div></td>
    <td>
</thead>
</table>

## Use ToMe to accelerate your training and inference

We support **[tomesd](https://github.com/dbolya/tomesd)** now! It is developed for stable-diffusion-based models referring to [ToMe](https://github.com/facebookresearch/ToMe), an efficient ViT speed-up tool based on token merging. To work on with **tomesd** in `mmagic`, you just need to add `tomesd_cfg` to `model` in [DreamBooth](./dreambooth.py). The only requirement is `torch >= 1.12.1` in order to properly support `torch.Tensor.scatter_reduce()` functionality. Please do check it before running the demo.

```python
model = dict(
    type='DreamBooth',
    ...
    tomesd_cfg=dict(ratio=0.5),
    ...
    val_prompts=val_prompts)
```

For more details, you can refer to [Stable Diffusion Acceleration](../stable_diffusion/README.md#use-tome-to-accelerate-your-stable-diffusion-model).

## Comments

Our codebase for the stable diffusion models builds heavily on [diffusers codebase](https://github.com/huggingface/diffusers) and the model weights are from [stable-diffusion-1.5](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py).

Thanks for the efforts of the community!

## Citation

```bibtex
@article{ruiz2022dreambooth,
  title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  journal={arXiv preprint arXiv:2208.12242},
  year={2022}
}
```
