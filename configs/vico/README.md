# ViCo (2023)

> [ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation](https://arxiv.org/abs/2208.12242)

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

|                                     Model                                      | Dataset | Download |
| :----------------------------------------------------------------------------: | :-----: | :------: |
|                         [ViCo](./vico.py)                          |    -    |    -     |

## Quick Start

1. Download [data](https://drive.google.com/drive/folders/1m8TCsY-C1tIOflHtWnFzTbw2C6dq67mC) and save to `data/vico/`

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
```

2. Start training with the following command:

```bash
bash tools/dist_train.sh configs/vico/vico.py 1
```

<!-- <table align="center">
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
</table> -->

## Use ToMe to accelerate your training and inference

We support **[tomesd](https://github.com/dbolya/tomesd)** now! It is developed for stable-diffusion-based models referring to [ToMe](https://github.com/facebookresearch/ToMe), an efficient ViT speed-up tool based on token merging. To work on with **tomesd** in `mmagic`, you just need to add `tomesd_cfg` to `model` in [DreamBooth](./dreambooth.py). The only requirement is `torch >= 1.12.1` in order to properly support `torch.Tensor.scatter_reduce()` functionality. Please do check it before running the demo.

```python
model = dict(
    type='ViCo',
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
@inproceedings{Hao2023ViCo,
  title={ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation},
  author={Shaozhe Hao and Kai Han and Shihao Zhao and Kwan-Yee K. Wong},
  year={2023}
}
```
