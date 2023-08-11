# Tutorial 3: Inference with pre-trained models

MMagic provides Hign-level APIs for you to easily play with state-of-the-art models on your own images or videos.

In the new API, only two lines of code are needed to implement inference:

```python
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance
editor = MMagicInferencer('pix2pix')
# Infer a image. Input image path and output image path is needed.
results = editor.infer(img='../resources/input/translation/gt_mask_0.png', result_out_dir='../resources/output/translation/tutorial_translation_pix2pix_res.jpg')
```

MMagic supports various fundamental generative models, including:

unconditional Generative Adversarial Networks (GANs), conditional GANs, diffusion models, etc.

MMagic also supports various applications, including:

text-to-image, image-to-image translation, 3D-aware generation, image super-resolution, video super-resolution, video frame interpolation, image inpainting, image matting, image restoration, image colorization, image generation, etc.

In this section, we will specify how to play with our pre-trained models.

- [Tutorial 3: Inference with Pre-trained Models](#tutorial-3-inference-with-pre-trained-models)
  - [Prepare some images or videos for inference](#Prepare-some-images-or-videos-for-inference)
  - [Generative Models](#Generative-Models)
    - [Unconditional Generative Adversarial Networks (GANs)](<#Unconditional-Generative-Adversarial-Networks-(GANs)>)
    - [Conditional Generative Adversarial Networks (GANs)](<#Conditional-Generative-Adversarial-Networks-(GANs)>)
    - [Diffusion Models](#Diffusion-Models)
  - [Applications](#Applications)
    - [Text-to-Image](#Text-to-Image)
    - [Image-to-image translation](#Image-to-image-translation)
    - [3D-aware generation](#3D-aware-generation)
    - [Image super-resolution](#Image-super-resolution)
    - [Video super-resolution](#Video-super-resolution)
    - [Video frame interpolation](Video-frame-interpolation)
    - [Image inpainting](#Image-inpainting)
    - [Image matting](#Image-matting)
    - [Image restoration](#Image-restoration)
    - [Image colorization](#Image-colorization)
- [Previous Versions](#Previous-Versions)

## Prepare some images or videos for inference

Please refer to our [tutorials](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_tutorial.ipynb) for details.

## Generative Models

### Unconditional Generative Adversarial Networks (GANs)

MMagic provides high-level APIs for sampling images with unconditional GANs. Unconditional GAN models do not need input, and output a image. We take 'styleganv1' as an example.

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
result_out_dir = './resources/output/unconditional/tutorial_unconditional_styleganv1_res.png'
editor = MMagicInferencer('styleganv1')
results = editor.infer(result_out_dir=result_out_dir)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir demo_unconditional_styleganv1_res.jpg
```

### Conditional Generative Adversarial Networks (GANs)

MMagic provides high-level APIs for sampling images with conditional GANs. Conditional GAN models take a label as input and output a image. We take 'biggan' as an example..

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
result_out_dir = './resources/output/conditional/tutorial_conditinal_biggan_res.jpg'
editor = MMagicInferencer('biggan', model_setting=1)
results = editor.infer(label=1, result_out_dir=result_out_dir)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name biggan \
        --model-setting 1 \
        --label 1 \
        --result-out-dir demo_conditional_biggan_res.jpg
```

### Diffusion Models

MMagic provides high-level APIs for sampling images with diffusion models. f

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
editor = MMagicInferencer(model_name='stable_diffusion')
text_prompts = 'A panda is having dinner at KFC'
result_out_dir = './resources/output/text2image/tutorial_text2image_sd_res.png'
editor.infer(text=text_prompts, result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name stable_diffusion \
        --text "A panda is having dinner at KFC" \
        --result-out-dir demo_text2image_stable_diffusion_res.png
```

## Applications

### Text-to-Image

Text-to-image models take text as input, and output a image. We take 'controlnet-canny' as an example.

```python
import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

control_url = 'https://user-images.githubusercontent.com/28132635/230288866-99603172-04cb-47b3-8adb-d1aa532d1d2c.jpg'
control_img = mmcv.imread(control_url)
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)

prompt = 'Room with blue walls and a yellow ceiling.'

output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 1 \
        --text "Room with blue walls and a yellow ceiling." \
        --control 'https://user-images.githubusercontent.com/28132635/230297033-4f5c32df-365c-4cf4-8e4f-1b76a4cbb0b7.png' \
        --result-out-dir demo_text2image_controlnet_canny_res.png
```

### Image-to-image translation

MMagic provides high-level APIs for translating images by using image translation models. Here is an example of building Pix2Pix and obtaining the translated images.

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
editor = MMagicInferencer('pix2pix')
results = editor.infer(img=img_path, result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name pix2pix \
        --img ${IMAGE_PATH} \
        --result-out-dir ${SAVE_PATH}
```

### 3D-aware generation

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
result_out_dir = './resources/output/eg3d-output'
editor = MMagicInferencer('eg3d')
results = editor.infer(result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
    --model-name eg3d \
    --result-out-dir ./resources/output/eg3d-output
```

### Image super-resolution

Image super resolution models take a image as input, and output a high resolution image. We take 'esrgan' as an example.

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
img = './resources/input/restoration/0901x2.png'
result_out_dir = './resources/output/restoration/tutorial_restoration_esrgan_res.png'
editor = MMagicInferencer('esrgan')
results = editor.infer(img=img, result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name esrgan \
        --img ${IMAGE_PATH} \
        --result-out-dir ${SAVE_PATH}
```

### Video super-resolution

```python
import os
from mmagic.apis import MMagicInferencer
from mmengine import mkdir_or_exist

# Create a MMagicInferencer instance and infer
video = './resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4'
result_out_dir = './resources/output/video_super_resolution/tutorial_video_super_resolution_basicvsr_res.mp4'
mkdir_or_exist(os.path.dirname(result_out_dir))
editor = MMagicInferencer('basicvsr')
results = editor.infer(video=video, result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name basicvsr \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_basicvsr_res.mp4
```

### Video frame interpolation

Video interpolation models take a video as input, and output a interpolated video. We take 'flavr' as an example.

```python
import os
from mmagic.apis import MMagicInferencer
from mmengine import mkdir_or_exist

# Create a MMagicInferencer instance and infer
video = './resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4'
result_out_dir = './resources/output/video_interpolation/tutorial_video_interpolation_flavr_res.mp4'
mkdir_or_exist(os.path.dirname(result_out_dir))
editor = MMagicInferencer('flavr')
results = editor.infer(video=video, result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name flavr \
        --video ${VIDEO_PATH} \
        --result-out-dir ${SAVE_PATH}
```

### Image inpainting

Inpaiting models take a masked image and mask pair as input, and output a inpainted image. We take 'global_local' as an example.

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

img = './resources/input/inpainting/celeba_test.png'
mask = './resources/input/inpainting/bbox_mask.png'

# Create a MMagicInferencer instance and infer
result_out_dir = './resources/output/inpainting/tutorial_inpainting_global_local_res.jpg'
editor = MMagicInferencer('global_local', model_setting=1)
results = editor.infer(img=img, mask=mask, result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name global_local  \
        --img ./resources/input/inpainting/celeba_test.png \
        --mask ./resources/input/inpainting/bbox_mask.png \
        --result-out-dir ./resources/output/inpainting/demo_inpainting_global_local_res.jpg
```

### Image matting

Inpaiting models take a image and trimap pair as input, and output a alpha image. We take 'gca' as an example.

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

img = './resources/input/matting/GT05.jpg'
trimap = './resources/input/matting/GT05_trimap.jpg'

# Create a MMagicInferencer instance and infer
result_out_dir = './resources/output/matting/tutorial_matting_gca_res.png'
editor = MMagicInferencer('gca')
results = editor.infer(img=img, trimap=trimap, result_out_dir=result_out_dir)
```

Use [demo/mmagic_inference_demo.py](../../../demo/mmagic_inference_demo.py) with the following commands:

```shell
python demo/mmagic_inference_demo.py \
        --model-name gca  \
        --img ./resources/input/matting/GT05.jpg \
        --trimap ./resources/input/matting/GT05_trimap.jpg \
        --result-out-dir ./resources/output/matting/demo_matting_gca_res.png
```

### Image restoration

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
img = './resources/input/restoration/0901x2.png'
result_out_dir = './resources/output/restoration/tutorial_restoration_nafnet_res.png'
editor = MMagicInferencer('nafnet')
results = editor.infer(img=img, result_out_dir=result_out_dir)
```

```shell
python demo/mmagic_inference_demo.py \
        --model-name nafnet \
        --img ./resources/input/restoration/0901x2.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_nafnet_res.png
```

### Image colorization

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
img = 'https://github-production-user-asset-6210df.s3.amazonaws.com/49083766/245713512-de973677-2be8-4915-911f-fab90bb17c40.jpg'
result_out_dir = './resources/output/colorization/tutorial_colorization_res.png'
editor = MMagicInferencer('inst_colorization')
results = editor.infer(img=img, result_out_dir=result_out_dir)
```

```shell
python demo/mmagic_inference_demo.py \
        --model-name inst_colorization \
        --img https://github-production-user-asset-6210df.s3.amazonaws.com/49083766/245713512-de973677-2be8-4915-911f-fab90bb17c40.jpg \
        --result-out-dir demo_colorization_res.png
```

## Previous Versions

If you want to use deprecated demos, please use [MMagic v1.0.0rc7](https://github.com/open-mmlab/mmagic/tree/v1.0.0rc7) and reference the [old tutorial](https://github.com/open-mmlab/mmagic/blob/v1.0.0rc7/docs/en/user_guides/inference.md).
