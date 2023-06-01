# MMagic Demo

There are some mmagic demos in this folder. We provide python command line usage here to run these demos and more guidance could also be found in the [documentation](https://mmagic.readthedocs.io/en/latest/user_guides/3_inference.html)

Table of contents:

[1. Download sample images or videos](#1-download-sample-images-or-videos)

[2. MMagic inference demo](#2-mmagic-inference-demo)

&#8195;    [2.1. Check supported tasks and models](#21-check-supported-tasks-and-models)

&#8195;    [2.2. Perform inference with command line](#22-perform-inference-with-command-line)

&#8195;      [2.2.1. Conditional GANs example](#221-conditional-gans)

&#8195;      [2.2.2. Inpainting example](#222-inpainting)

&#8195;      [2.2.3. Matting example](#223-matting)

&#8195;      [2.2.4. Image Super-Resolution example](#224-image-super-resolution)

&#8195;      [2.2.5. Image Translation example](#225-image-translation)

&#8195;      [2.2.6. Unconditional GANs example](#226-unconditional-gans)

&#8195;      [2.2.7. Video Interpolation example](#227-video-interpolation)

&#8195;      [2.2.8. Video Super-Resolution example](#228-video-super-resolution)

&#8195;      [2.2.9. Text-to-Image example](#229-text-to-image)

&#8195;      [2.2.10. 3D-aware Generation example](#2210-3d-aware-generation)

&#8195;      [2.2.11. Image Restoration example](#2211-image-restoration)

[3. Other demos](#3-other-demos)

## 1. Download sample images or videos

We prepared some images and videos for you to run demo with. After MMagic is well installed, you could use demos in this folder to infer these data.
Download with python script [download_inference_resources.py](./download_inference_resources.py).

```shell
# cd mmagic demo path
cd mmagic/demo

# see all resources
python download_inference_resources.py --print-all
# see all task types
python download_inference_resources.py --print-task-type
# see resources of one specific task
python download_inference_resources.py --print-task 'Inpainting'
# download all resources to default dir '../resources'
python download_inference_resources.py
# download resources of one task
python download_inference_resources.py --task 'Inpainting'
# download to the directory you want
python download_inference_resources.py --root-dir '../resources'
```

## 2. MMagic inference demo

### 2.1 Check supported tasks and models

print all supported models for inference.

```shell
python mmagic_inference_demo.py --print-supported-models
```

print all supported tasks for inference.

```shell
python mmagic_inference_demo.py --print-supported-tasks
```

print all supported models for one task, take 'Image2Image' for example.

```shell
python mmagic_inference_demo.py --print-task-supported-models 'Image2Image'
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

#### 2.2.1 Conditional GANs

```shell
python mmagic_inference_demo.py \
        --model-name biggan \
        --label 1 \
        --result-out-dir ../resources/output/conditional/demo_conditional_biggan_res.jpg
```

#### 2.2.2 Inpainting

```shell
python mmagic_inference_demo.py \
        --model-name global_local  \
        --img ../resources/input/inpainting/celeba_test.png \
        --mask ../resources/input/inpainting/bbox_mask.png \
        --result-out-dir ../../resources/output/inpainting/demo_inpainting_global_local_res.jpg
```

#### 2.2.3 Matting

```shell
python mmagic_inference_demo.py \
        --model-name gca  \
        --img ../resources/input/matting/GT05.jpg \
        --trimap ../resources/input/matting/GT05_trimap.jpg \
        --result-out-dir ../resources/output/matting/demo_matting_gca_res.png
```

#### 2.2.4 Image Super-resolution

```shell
python mmagic_inference_demo.py \
        --model-name esrgan \
        --img ../resources/input/restoration/0901x2.png \
        --result-out-dir ../resources/output/restoration/demo_restoration_esrgan_res.png
```

```shell
python mmagic_inference_demo.py \
        --model-name ttsr \
        --img ../resources/input/restoration/0901x2.png \
        --ref ../resources/input/restoration/0901x2.png \
        --result-out-dir ../resources/output/restoration/demo_restoration_ttsr_res.png
```

#### 2.2.5 Image translation

```shell
python mmagic_inference_demo.py \
        --model-name pix2pix \
        --img ../resources/input/translation/gt_mask_0.png \
        --result-out-dir ../resources/output/translation/demo_translation_pix2pix_res.png
```

#### 2.2.6 Unconditional GANs

```shell
python mmagic_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir ../resources/output/unconditional/demo_unconditional_styleganv1_res.jpg
```

#### 2.2.7 Video interpolation

```shell
python mmagic_inference_demo.py \
        --model-name flavr \
        --video ../resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4 \
        --result-out-dir ../resources/output/video_interpolation/demo_video_interpolation_flavr_res.mp4
```

#### 2.2.8 Video Super-Resolution

BasicVSR / BasicVSR++ / IconVSR / RealBasicVSR

```shell
python mmagic_inference_demo.py \
        --model-name basicvsr \
        --video ../resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ../resources/output/video_restoration/demo_video_restoration_basicvsr_res.mp4
```

EDVR

```shell
python mmagic_inference_demo.py \
        --model-name edvr \
        --extra-parameters window_size=5 \
        --video ../resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ../resources/output/video_restoration/demo_video_restoration_edvr_res.mp4
```

TDAN

```shell
python mmagic_inference_demo.py \
        --model-name tdan \
        --model-setting 2
        --extra-parameters window_size=5 \
        --video ../resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ../resources/output/video_restoration/demo_video_restoration_edvr_res.mp4
```

#### 2.2.9 Text-to-Image

```shell
python mmagic_inference_demo.py \
        --model-name stable_diffusion \
        --text "A panda is having dinner at KFC" \
        --result-out-dir ../resources/output/text2image/demo_text2image_stable_diffusion_res.png
```

#### 2.2.10 3D-aware Generation

```shell
python demo/mmagic_inference_demo.py \
    --model-name eg3d \
    --result-out-dir ../resources/output/eg3d-output
```

#### 2.2.11 Image Restoration

```shell
python mmagic_inference_demo.py \
        --model-name nafnet \
        --img ../resources/input/restoration/0901x2.png \
        --result-out-dir ../resources/output/restoration/demo_restoration_nafnet_res.png
```
