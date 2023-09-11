# Diffusers Pipeline (2023)

> [Diffusers Pipeline](https://github.com/huggingface/diffusers)

> **Task**: Diffusers Pipeline

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

For the convenience of our community users, this inferencer supports using the pipelines from diffusers for inference to compare the results with the algorithms supported within our algorithm library.

## Configs

|                   Model                   | Dataset | Download |
| :---------------------------------------: | :-----: | :------: |
| [diffusers pipeline](./sd_xl_pipeline.py) |    -    |    -     |

## Quick Start

### sd_xl_pipeline

To run stable diffusion XL with mmagic inference API, follow these codes:

```python
from mmagic.apis import MMagicInferencer

# Create a MMEdit instance and infer
editor = MMagicInferencer(model_name='diffusers_pipeline')
text_prompts = 'Japanese anime style, girl, beautiful, cute, colorful, best quality, extremely detailed'
negative_prompt = 'bad face, bad hands'
result_out_dir = 'sd_xl_japanese.png'
editor.infer(text=text_prompts,
             negative_prompt=negative_prompt,
             result_out_dir=result_out_dir)
```

You will get this picture:

<div align=center >
 <img src="https://user-images.githubusercontent.com/12782558/266557074-53519887-6597-42cf-8a0b-03c2db3f4ab2.png" width="600"/>
</div >

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
