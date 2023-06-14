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
