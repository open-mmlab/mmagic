# Diffusers Pipeline (2023)

> [Diffusers Pipeline](https://github.com/huggingface/diffusers)

> **Task**: Diffusers Pipeline

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We support diffusers pipelines for users to conveniently use diffusers to do inferece in our repo.

## Configs

|                   Model                   | Dataset | Download |
| :---------------------------------------: | :-----: | :------: |
| [diffusers pipeline](./sd_xl_pipeline.py) |    -    |    -     |

## Quick Start

```python
from mmagic.apis import MMagicInferencer

# Create a MMEdit instance and infer
editor = MMagicInferencer(model_name='diffusers_pipeline')
text_prompts = 'Japanese anime style, girl, beautiful, cute, colorful, best quality, extremely detailed'
negative_prompt = 'bad face, bad hands'
result_out_dir = 'resources/output/text2image/sd_xl_japanese.png'
editor.infer(text=text_prompts,
             negative_prompt=negative_prompt,
             result_out_dir=result_out_dir)
```

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
