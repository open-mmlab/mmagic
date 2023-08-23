# 教程3：使用预训练模型推理

MMagic 提供了高级API，让您可以轻松地在自己的图像或视频上使用最先进的模型进行操作。
在新的API中，仅需两行代码即可进行推理。

```python
from mmagic.apis import MMagicInferencer

# 创建MMagicInferencer实例
editor = MMagicInferencer('pix2pix')
# 推理图片.需要输入图片路径与输出图片路径
results = editor.infer(img='../resources/input/translation/gt_mask_0.png', result_out_dir='../resources/output/translation/tutorial_translation_pix2pix_res.jpg')
```

MMagic支持各种基础生成模型，包括无条件生成对抗网络（GANs）、条件GANs、扩散模型等。
MMagic 同样支持多种应用，包括：文生图、图生图的转换、3D感知生成、图像超分、视频超分、视频帧插值、图像修补、图像抠图、图像恢复、图像上色、图像生成等。
在本节中，我们将详细说明如何使用我们预训练的模型进行操作。

- 教程3: 使用预训练模型推理
  - [准备一些图片或者视频用于推理](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Prepare-some-images-or-videos-for-inference)
  - 生成模型
    - [无条件生成对抗网络 (GANs)](<https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Unconditional-Generative-Adversarial-Networks-(GANs)>)
    - [条件生成对抗网络(GANs)](<https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Conditional-Generative-Adversarial-Networks-(GANs)>)
    - [扩散模型](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Diffusion-Models)
  - 应用
    - [文生图](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Text-to-Image)
    - [图生图的转换](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Image-to-image-translation)
    - [3D感知生成](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#3D-aware-generation)
    - [图像超分](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Image-super-resolution)
    - [视频超分](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Video-super-resolution)
    - [视频帧插值](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/Video-frame-interpolation)
    - [图像修补](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Image-inpainting)
    - [图像抠图](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Image-matting)
    - [图像恢复](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Image-restoration)
    - [图像上色](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Image-colorization)
- [以前的版本](https://github.com/open-mmlab/mmagic/blob/main/docs/zh_cn/user_guides/inference.md#Previous-Versions)

## 准备一些图片或者视频用于推理

请参考我们的[教程](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_tutorial.ipynb)获取详细信息。

## 生成模型

### 无条件生成对抗网络（GANs）

MMagic提供了用于使用无条件GANs进行图像采样的高级API。无条件GAN模型不需要输入，并输出一张图像。我们以'styleganv1'为例。

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# 创建MMagicInferencer实例，并进行推理
result_out_dir = './resources/output/unconditional/tutorial_unconditional_styleganv1_res.png'
editor = MMagicInferencer('styleganv1')
results = editor.infer(result_out_dir=result_out_dir)
```

确实，我们已经为用户提供了一个更友好的演示脚本。您可以使用以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)：

```python
python demo/mmagic_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir demo_unconditional_styleganv1_res.jpg
```

### 条件生成对抗网络(GANs)

MMagic提供了使用条件GAN进行图像采样的高级API。条件GAN模型接受一个标签作为输入，并输出一张图像。我们以'biggan'为例。

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# 创建MMagicInferencer实例，并进行推理
result_out_dir = './resources/output/conditional/tutorial_conditinal_biggan_res.jpg'
editor = MMagicInferencer('biggan', model_setting=1)
results = editor.infer(label=1, result_out_dir=result_out_dir)
```

我们已经为用户提供了一个更友好的演示脚本。您可以使用以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)：

```shell
python demo/mmagic_inference_demo.py \
        --model-name biggan \
        --model-setting 1 \
        --label 1 \
        --result-out-dir demo_conditional_biggan_res.jpg
```

### 扩散模型

MMagic提供了使用扩散模型进行图像采样的高级API。

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# 创建MMagicInferencer实例，并进行推理
editor = MMagicInferencer(model_name='stable_diffusion')
text_prompts = 'A panda is having dinner at KFC'
result_out_dir = './resources/output/text2image/tutorial_text2image_sd_res.png'
editor.infer(text=text_prompts, result_out_dir=result_out_dir)
```

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name stable_diffusion \
        --text "A panda is having dinner at KFC" \
        --result-out-dir demo_text2image_stable_diffusion_res.png
```

## 应用

### 文生图

文生图模型将文本作为输入，输出一张图片。我们以'controlnet-canny'为例。

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

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 1 \
        --text "Room with blue walls and a yellow ceiling." \
        --control 'https://user-images.githubusercontent.com/28132635/230297033-4f5c32df-365c-4cf4-8e4f-1b76a4cbb0b7.png' \
        --result-out-dir demo_text2image_controlnet_canny_res.png
```

### 图生图的转换

MMagic提供了使用图像翻译模型进行图像翻译的高级API。下面是构建Pix2Pix并获取翻译图像的示例。

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
editor = MMagicInferencer('pix2pix')
results = editor.infer(img=img_path, result_out_dir=result_out_dir)
```

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name pix2pix \
        --img ${IMAGE_PATH} \
        --result-out-dir ${SAVE_PATH}
```

### 3D感知生成

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

# Create a MMagicInferencer instance and infer
result_out_dir = './resources/output/eg3d-output'
editor = MMagicInferencer('eg3d')
results = editor.infer(result_out_dir=result_out_dir)
```

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
    --model-name eg3d \
    --result-out-dir ./resources/output/eg3d-output
```

### 图像超分

图像超分辨率模型接受一张图像作为输入，并输出一张高分辨率图像。我们以 'esrgan' 为例。

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

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name esrgan \
        --img ${IMAGE_PATH} \
        --result-out-dir ${SAVE_PATH}
```

### 视频超分

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

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name basicvsr \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_basicvsr_res.mp4
```

### 视频帧插值

视频插值模型接受一个视频作为输入，并输出一个插值后的视频。我们以 'flavr' 为例。

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

### 图像修补

修复模型接受一对屏蔽图像和屏蔽蒙版作为输入，并输出一个修复后的图像。我们以 'global_local' 为例。

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

img = './resources/input/matting/GT05.jpg'
trimap = './resources/input/matting/GT05_trimap.jpg'

# 创建MMagicInferencer实例，并进行推理
result_out_dir = './resources/output/matting/tutorial_matting_gca_res.png'
editor = MMagicInferencer('gca')
results = editor.infer(img=img, trimap=trimap, result_out_dir=result_out_dir)
```

### 图像抠图

**抠图**模型接受一对图像和修剪映射作为输入，并输出一个 alpha 图像。我们以 'gca' 为例。

```python
import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer

img = './resources/input/matting/GT05.jpg'
trimap = './resources/input/matting/GT05_trimap.jpg'

# 创建MMagicInferencer实例，并进行推理
result_out_dir = './resources/output/matting/tutorial_matting_gca_res.png'
editor = MMagicInferencer('gca')
results = editor.infer(img=img, trimap=trimap, result_out_dir=result_out_dir)
```

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name gca  \
        --img ./resources/input/matting/GT05.jpg \
        --trimap ./resources/input/matting/GT05_trimap.jpg \
        --result-out-dir ./resources/output/matting/demo_matting_gca_res.png
```

### 图像恢复

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

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name nafnet \
        --img ./resources/input/restoration/0901x2.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_nafnet_res.png
```

### 图像上色

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

通过以下命令使用[demo/mmagic_inference_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/mmagic_inference_demo.py)

```shell
python demo/mmagic_inference_demo.py \
        --model-name inst_colorization \
        --img https://github-production-user-asset-6210df.s3.amazonaws.com/49083766/245713512-de973677-2be8-4915-911f-fab90bb17c40.jpg \
        --result-out-dir demo_colorization_res.png
```

## 以前的版本

如果您想使用已弃用的演示，请使用[MMagic v1.0.0rc7](https://github.com/open-mmlab/mmagic/tree/v1.0.0rc7)并参考[旧教程](https://github.com/open-mmlab/mmagic/blob/v1.0.0rc7/docs/en/user_guides/inference.md)。
