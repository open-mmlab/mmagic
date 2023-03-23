# Controlnet Animation (2023)

> [Controlnet](https://github.com/lllyasviel/ControlNet) Application

> **Task**: controlnet_animation

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

ControlNet is a neural network structure to control diffusion models by adding extra conditions.
We use controlnet to generate frames guided by input video and make animation.

## Demos

prompt key words: a handsome man, silver hair, smiling, play basketball

<div align="center">
  <video src="https://user-images.githubusercontent.com/12782558/227149757-fd054d32-554f-45d5-9f09-319184866d85.mp4" width=1024/>
</div>

prompt key words: a handsome man

<div align="center">
  <video src="https://user-images.githubusercontent.com/12782558/227152129-d70d5f76-a6fc-4d23-97d1-a94abd08f95a.mp4" width=1024/>
</div>

**Change prompt to get different result video**

prompt key words: a girl, black hair, white pants, smiling, play basketball

<div align="center">
  <video src="https://user-images.githubusercontent.com/12782558/227216038-38599164-2384-4a79-b65e-f98785d466bf.mp4" width=512/>
</div>

## Pretrained models

We use pretrained model from hugging face.

|                Model                 | Dataset | Download |
| :----------------------------------: | :-----: | :------: |
| [anythingv3](./anythingv3_config.py) |    -    |    -     |

## Quick Start

Running the following codes, you can get a text-generated image.

```python
from mmedit.edit import MMEdit

# Create a MMEdit instance and infer
editor = MMEdit(model_name='controlnet_animation')

prompt = 'a girl, black hair, T-shirt, ' + \
         'smoking, best quality, extremely detailed'
negative_prompt = 'longbody, lowres, bad anatomy, ' + \
                  'bad hands, missing fingers, extra digit, ' + \
                  'fewer digits, cropped, worst quality, low quality'
video = '/path/to/your/input/video.mp4'
save_path = '/path/to/your/output/video.mp4'

editor.infer(
    video=video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_path=save_path)
```

## Citation

```bibtex
@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Lvmin Zhang and Maneesh Agrawala},
  year={2023},
  eprint={2302.05543},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
