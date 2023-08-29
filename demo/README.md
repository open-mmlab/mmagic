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
python mmediting_inference_demo.py --print-supported-models
```

print all supported tasks for inference.

```shell
python mmediting_inference_demo.py --print-supported-tasks
```

print all supported models for one task, take 'Image2Image' for example.

```shell
python mmediting_inference_demo.py --print-task-supported-models 'Image2Image'
```

### 2.2 Perform inference with command line

You can use the following commands to perform inference with a MMagic model.

Usage of python API can also be found in this [tutotial](./mmediting_inference_tutorial.ipynb).

```shell
python demo/mmediting_inference_demo.py \
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
python mmediting_inference_demo.py \
        --model-name biggan \
        --model-setting 3 \
        --label 1 \
        --result-out-dir demo_conditional_biggan_res.jpg
```

#### 2.2.3 Unconditional GANs

```shell
python mmediting_inference_demo.py \
        --model-name global_local  \
        --img ../resources/input/inpainting/celeba_test.png \
        --mask ../resources/input/inpainting/bbox_mask.png \
        --result-out-dir ../../resources/output/inpainting/demo_inpainting_global_local_res.jpg
```

#### 2.2.4 Image Translation

```shell
python mmediting_inference_demo.py \
        --model-name global_local  \
        --img ../resources/input/matting/GT05.jpg \
        --mask ../resources/input/matting/GT05_trimap.jpg \
        --result-out-dir ../resources/output/matting/demo_matting_gca_res.png
```

#### 2.2.5 Inpainting

```shell
python mmediting_inference_demo.py \
        --model-name esrgan \
        --img ../resources/input/restoration/0901x2.png \
        --result-out-dir ../resources/output/restoration/demo_restoration_esrgan_res.png
```

#### 2.2.6 Matting

```shell
python mmediting_inference_demo.py \
        --model-name pix2pix \
        --img ../resources/input/translation/gt_mask_0.png \
        --result-out-dir ../resources/output/translation/demo_translation_pix2pix_res.png
```

#### 2.2.7 Image Restoration

```shell
python mmediting_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir ../resources/output/unconditional/demo_unconditional_styleganv1_res.jpg
```

#### 2.2.8 Image Super-resolution

```shell
python mmediting_inference_demo.py \
        --model-name flavr \
        --video ../resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4 \
        --result-out-dir ../resources/output/video_interpolation/demo_video_interpolation_flavr_res.mp4
```

EDVR

```shell
python mmediting_inference_demo.py \
        --model-name edvr \
        --extra-parameters window_size=5 \
        --video ./resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ./resources/output/video_restoration/demo_video_restoration_edvr_res.mp4
```

TDAN

```shell
python mmediting_inference_demo.py \
        --model-name disco \
        --text 0=["clouds surround the mountains and Chinese palaces,sunshine,lake,overlook,overlook,unreal engine,light effect,Dream，Greg Rutkowski,James Gurney,artstation"] \
        --result-out-dir ../resources/output/text2image/demo_text2image_disco_res.png
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
python demo/mmediting_inference_demo.py \
    --model-name eg3d \
    --result-out-dir ./resources/output/eg3d-output
```

## 3. Other demos

## 3.1 gradio demo

#### 3.1.1 DragGAN

First, put your checkpoint path in `./checkpoints`, *e.g.* `./checkpoints/stylegan2_lions_512_pytorch_mmagic.pth`

Then, try on the script:

```shell
python demo/gradio_draggan.py
```
