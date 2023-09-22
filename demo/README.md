# MMagic Demo

There are some mmagic demos in this folder. We provide python command line usage here to run these demos and more guidance could also be found in the [documentation](https://mmagic.readthedocs.io/en/latest/user_guides/inference.html)

Table of contents:

[1. Download sample images or videos](#1-download-sample-images-or-videos)

[2. MMagic inference demo](#2-mmagic-inference-demo)

&#8195;    [2.1. Check supported tasks and models](#21-check-supported-tasks-and-models)

&#8195;    [2.2. Perform inference with command line](#22-perform-inference-with-command-line)

&#8195;      [2.2.1. Text-to-Image example](#221-text-to-image)

&#8195;      [2.2.2. Conditional GANs example](#222-conditional-gans)

&#8195;      [2.2.3. Unconditional GANs example](#223-unconditional-gans)

&#8195;      [2.2.4. Image Translation (Image2Image) example](#224-image-translation)

&#8195;      [2.2.5. Inpainting example](#225-inpainting)

&#8195;      [2.2.6. Matting example](#226-matting)

&#8195;      [2.2.7. Image Restoration example](#227-image-restoration)

&#8195;      [2.2.8. Image Super-Resolution example](#228-image-super-resolution)

&#8195;      [2.2.9. Video Super-Resolution example](#229-video-super-resolution)

&#8195;      [2.2.10. Video Interpolation example](#2210-video-interpolation)

&#8195;      [2.2.11. Image Colorization example](#2211-image-colorization)

&#8195;      [2.2.12. 3D-aware Generation example](#2212-3d-aware-generation)

[3. Other demos](#3-other-demos)

&#8195;    [3.1. Gradio demo](#31-gradio-demo)

&#8195;      [3.1.1. DragGAN](#311-draggan)

&#8195;      [3.1.2. ViCo](#312-vico)

&#8195;      [3.1.3. FastComposer](#313-fastcomposer)

&#8195;      [3.1.4. AnimateDiff](#314-animatediff)

## 1. Download sample images or videos

We prepared some images and videos for you to run demo with. After MMagic is well installed, you could use demos in this folder to infer these data.
Download with python script [download_inference_resources.py](./download_inference_resources.py).

```shell
# see all resources
python demo/download_inference_resources.py --print-all

# see all task types
python demo/download_inference_resources.py --print-task-type

# see resources of one specific task
python demo/download_inference_resources.py --print-task 'Inpainting'

# download all resources to default dir './resources'
python demo/download_inference_resources.py

# download resources of one task
python demo/download_inference_resources.py --task 'Inpainting'

# download to the directory you want
python demo/download_inference_resources.py --root-dir './resources'
```

## 2. MMagic inference demo

### 2.1 Check supported tasks and models

print all supported models for inference.

```shell
python demo/mmagic_inference_demo.py --print-supported-models
```

print all supported tasks for inference.

```shell
python demo/mmagic_inference_demo.py --print-supported-tasks
```

print all supported models for one task, take 'Image2Image' for example.

```shell
python demo/mmagic_inference_demo.py --print-task-supported-models 'Text2Image'
```

### 2.2 Perform inference with command line

You can use the following commands to perform inference with a MMagic model.

Usage of python API can also be found in this [tutotial](./mmagic_inference_tutorial.ipynb).

```shell
python demo/mmagic_inference_demo.py \
    [--img] \
    [--video] \
    [--label] \
    [--trimap] \
    [--mask] \
    [--result-out-dir] \
    [--model-name] \
    [--model-setting] \
    [--model-config] \
    [--model-ckpt] \
    [--device ] \
    [--extra-parameters]
```

Examples for each kind of task:

#### 2.2.1 Text-to-Image

stable diffusion

```shell
python demo/mmagic_inference_demo.py \
        --model-name stable_diffusion \
        --text "A panda is having dinner at KFC" \
        --result-out-dir demo_text2image_stable_diffusion_res.png
```

controlnet-canny

```shell
python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 1 \
        --text "Room with blue walls and a yellow ceiling." \
        --control 'https://user-images.githubusercontent.com/28132635/230297033-4f5c32df-365c-4cf4-8e4f-1b76a4cbb0b7.png' \
        --result-out-dir demo_text2image_controlnet_canny_res.png
```

controlnet-pose

```shell
python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 2 \
        --text "masterpiece, best quality, sky, black hair, skirt, sailor collar, looking at viewer, short hair, building, bangs, neckerchief, long sleeves, cloudy sky, power lines, shirt, cityscape, pleated skirt, scenery, blunt bangs, city, night, black sailor collar, closed mouth" \
        --control 'https://user-images.githubusercontent.com/28132635/230380893-2eae68af-d610-4f7f-aa68-c2f22c2abf7e.png' \
        --result-out-dir demo_text2image_controlnet_pose_res.png
```

controlnet-seg

```shell
python demo/mmagic_inference_demo.py \
        --model-name controlnet \
        --model-setting 3 \
        --text "black house, blue sky" \
        --control 'https://github-production-user-asset-6210df.s3.amazonaws.com/49083766/243599897-553a4c46-c61d-46df-b820-59a49aaf6678.png' \
        --result-out-dir demo_text2image_controlnet_seg_res.png
```

#### 2.2.2 Conditional GANs

```shell
python demo/mmagic_inference_demo.py \
        --model-name biggan \
        --model-setting 3 \
        --label 1 \
        --result-out-dir demo_conditional_biggan_res.jpg
```

#### 2.2.3 Unconditional GANs

```shell
python demo/mmagic_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir demo_unconditional_styleganv1_res.jpg
```

#### 2.2.4 Image Translation

```shell
python demo/mmagic_inference_demo.py \
        --model-name pix2pix \
        --img ./resources/input/translation/gt_mask_0.png \
        --result-out-dir ./resources/output/translation/demo_translation_pix2pix_res.png
```

#### 2.2.5 Inpainting

```shell
python demo/mmagic_inference_demo.py \
        --model-name deepfillv2  \
        --img ./resources/input/inpainting/celeba_test.png \
        --mask ./resources/input/inpainting/bbox_mask.png \
        --result-out-dir ./resources/output/inpainting/demo_inpainting_deepfillv2_res.jpg
```

#### 2.2.6 Matting

```shell
python demo/mmagic_inference_demo.py \
        --model-name aot_gan  \
        --img ./resources/input/matting/GT05.jpg \
        --trimap ./resources/input/matting/GT05_trimap.jpg \
        --result-out-dir ./resources/output/matting/demo_matting_gca_res.png
```

#### 2.2.7 Image Restoration

```shell
python demo/mmagic_inference_demo.py \
        --model-name nafnet \
        --img ./resources/input/restoration/0901x2.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_nafnet_res.png
```

#### 2.2.8 Image Super-resolution

```shell
python demo/mmagic_inference_demo.py \
        --model-name esrgan \
        --img ./resources/input/restoration/0901x2.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_esrgan_res.png
```

```shell
python demo/mmagic_inference_demo.py \
        --model-name ttsr \
        --img ./resources/input/restoration/000001.png \
        --ref ./resources/input/restoration/000001.png \
        --result-out-dir ./resources/output/restoration/demo_restoration_ttsr_res.png
```

#### 2.2.9 Video Super-Resolution

BasicVSR / BasicVSR++ / IconVSR / RealBasicVSR

```shell
python demo/mmagic_inference_demo.py \
        --model-name basicvsr \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_basicvsr_res.mp4
```

EDVR

```shell
python demo/mmagic_inference_demo.py \
        --model-name edvr \
        --extra-parameters window_size=5 \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_edvr_res.mp4
```

TDAN

```shell
python demo/mmagic_inference_demo.py \
        --model-name tdan \
        --model-setting 2 \
        --extra-parameters window_size=5 \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_tdan_res.mp4
```

#### 2.2.10 Video interpolation

```shell
python demo/mmagic_inference_demo.py \
        --model-name flavr \
        --video ./resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4 \
        --result-out-dir ./resources/output/video_interpolation/demo_video_interpolation_flavr_res.mp4
```

#### 2.2.11 Image Colorization

```
python demo/mmagic_inference_demo.py \
        --model-name inst_colorization \
        --img https://github-production-user-asset-6210df.s3.amazonaws.com/49083766/245713512-de973677-2be8-4915-911f-fab90bb17c40.jpg \
        --result-out-dir demo_colorization_res.png
```

#### 2.2.12 3D-aware Generation

```shell
python demo/mmagic_inference_demo.py \
    --model-name eg3d \
    --result-out-dir ./resources/output/eg3d-output
```

## 3. Other demos

## 3.1 gradio demo

#### 3.1.1 DragGAN

First, put your checkpoint path in `./checkpoints`, *e.g.* `./checkpoints/stylegan2_lions_512_pytorch_mmagic.pth`. For example,

```shell
mkdir checkpoints
cd checkpoints
wget -O stylegan2_lions_512_pytorch_mmagic.pth https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-Lions-internet
```

Then, try on the script:

```shell
python demo/gradio_draggan.py
```

#### 3.1.2 ViCo

Launch the UI.

```shell
python demo/gradio_vico.py
```

*Training*

1. Submit your concept sample images to the interface and fill in the *init_token* and *placeholder*.

2. Click the *Start Training* button.

3. Your training results will be under the folder `./work_dirs/vico_gradio`.

*Inference*

Follow the [instructions](../configs/vico/README.md#quick-start#4) to download the pretrained weights (or [use your own weights](../configs/vico/README.md#quick-start#5)) and put them under the folder `./ckpts`

```
mkdir ckpts
```

your folder structure should be like:

```
ckpts
└── barn.pth
└── batman.pth
└── clock.pth
...
```

Then launch the UI and you can use the pretrained weights to generate images.

1. Upload reference image.

2. (Optional) Customize advanced settings.

3. Click inference button.

#### 3.1.3 FastComposer

First, run the script:

```shell
python demo/gradio_fastcomposer.py
```

Second, upload reference subject images.For example,

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/14927720/265911400-91635451-54b6-4dc6-92a7-c1d02f88b62e.jpeg" width="400"/>
  <br/>
  <b>'reference_0.png'</b>
</div></td>
    <td>
<div align="center">
  <img src="https://user-images.githubusercontent.com/14927720/265911502-66b67f53-dff0-4d25-a9af-3330e446aa48.jpeg" width="400"/>
  <br/>
  <b>'reference_1.png'</b>
</div></td>
    <td>
</thead>
</table>

Then, add prompt like `A man img and a man img sitting together` and press `run` button.

Finally, you can get text-generated images.

<div align=center>
<img src="https://user-images.githubusercontent.com/14927720/265911526-4975d6e2-c5fc-4324-80c9-a7a512953218.png">
</div>

#### 3.1.4 AnimateDiff

1. Download [ToonYou](https://civitai.com/api/download/models/78775) and MotionModule checkpoint

```bash
#!/bin/bash

mkdir models && cd models
mkdir Motion_Module && mkdir DreamBooth_LoRA
gdown 1RqkQuGPaCO5sGZ6V6KZ-jUWmsRu48Kdq -O models/Motion_Module/
gdown 1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu -O models/Motion_Module/
wget https://civitai.com/api/download/models/78775 -P models/DreamBooth_LoRA/ --content-disposition --no-check-certificate
```

2. Modify the config file in `configs/animatediff/animatediff_ToonYou.py`

```python

models_path = '/home/AnimateDiff/models/'
```

3. Then, try on the script:

```shell
# may need to install imageio[ffmpeg]:
# pip install imageio-ffmpeg
python demo/gradio_animatediff.py
```

4. Select SD, MotionModule and DreamBooth checkpoints. Adjust inference parameters. Then input a selected prompt and its relative negative_prompt:

```python

prompts = [
    "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress",

    "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes,",

    "best quality, masterpiece, 1boy, formal, abstract, looking at viewer, masculine, marble pattern",

    "best quality, masterpiece, 1girl, cloudy sky, dandelion, contrapposto, alternate hairstyle,"
]
negative_prompts = [
    "",
    "badhandv4,easynegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3, bad-artist, bad_prompt_version2-neg, teeth",
    "",
    "",
]
# More test samples could be generated with other config files. Please check 'configs/animatediff/README.md'
```

5. Click the 'Generate' button.
