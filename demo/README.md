# MMEditing Demo

There are some mmediting demos in this folder. We provide python command line usage here to run these demos and more guidance could also be found in the [documentation](https://mmediting.readthedocs.io/en/dev-1.x/user_guides/3_inference.html)

Table of contents:

[1. Download sample images or videos](#1-download-sample-images-or-videos)

[2. MMEditing inference demo](#2-mmediting-inference-demo)

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

[3. Other demos](#3-other-demos)

## 1. Download sample images or videos

We prepared some images and videos for you to run demo with. After MMEdit is well installed, you could use demos in this folder to infer these data.
Download with python script [download_inference_resources.py](./download_inference_resources.py).

```shell
# cd mmediting demo path
cd mmediting/demo

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

## 2. MMEditing inference demo

### 2.1 Check supported tasks and models

print all supported models for inference.

```shell
python mmediting_inference_demo.py --print-supported-models
```

print all supported tasks for inference.

```shell
python mmediting_inference_demo.py --print-supported-tasks
```

print all supported models for one task, take 'Image2Image Translation' for example.

```shell
python mmediting_inference_demo.py --print-task-supported-models 'Image2Image Translation'
```

### 2.2 Perform inference with command line

You can use the following commands to perform inference with a MMEdit model.

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

#### 2.2.1 Conditional GANs

```shell
python mmediting_inference_demo.py \
        --model-name biggan \
        --label 1 \
        --result-out-dir ../resources/output/conditional/demo_conditional_biggan_res.jpg
```

#### 2.2.2 Inpainting

```shell
python mmediting_inference_demo.py \
        --model-name global_local  \
        --img ../resources/input/inpainting/celeba_test.png \
        --mask ../resources/input/inpainting/bbox_mask.png \
        --result-out-dir ../../resources/output/inpainting/demo_inpainting_global_local_res.jpg
```

#### 2.2.3 Matting

```shell
python mmediting_inference_demo.py \
        --model-name global_local  \
        --img ../resources/input/matting/GT05.jpg \
        --mask ../resources/input/matting/GT05_trimap.jpg \
        --result-out-dir ../resources/output/matting/demo_matting_gca_res.png
```

#### 2.2.4 Image Super-resolution

```shell
python mmediting_inference_demo.py \
        --model-name esrgan \
        --img ../resources/input/restoration/0901x2.png \
        --result-out-dir ../resources/output/restoration/demo_restoration_esrgan_res.png
```

#### 2.2.5 Image translation

```shell
python mmediting_inference_demo.py \
        --model-name pix2pix \
        --img ../resources/input/translation/gt_mask_0.png \
        --result-out-dir ../resources/output/translation/demo_translation_pix2pix_res.png
```

#### 2.2.6 Unconditional GANs

```shell
python mmediting_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir ../resources/output/unconditional/demo_unconditional_styleganv1_res.jpg
```

#### 2.2.7 Video interpolation

```shell
python mmediting_inference_demo.py \
        --model-name flavr \
        --video ../resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4 \
        --result-out-dir ../resources/output/video_interpolation/demo_video_interpolation_flavr_res.mp4
```

#### 2.2.8 Video Super-Resolution

```shell
python mmediting_inference_demo.py \
        --model-name edvr \
        --extra-parameters window_size=5 \
        --video ../resources/input/video_restoration/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir ../resources/output/video_restoration/demo_video_restoration_edvr_res.mp4
```

#### 2.2.9 EG3D

```shell
python demo/mmediting_inference_demo.py \
    --model-name eg3d \
    --model-config configs/eg3d/eg3d_cvt-official-rgb_shapenet-128x128.py \
    --model-ckpt shapenet_ema.pt \
    --result-out-dir eg3d_output \
    --interpolation both \
    --num-frames 50 \
    --vis-mode depth
```

## 3. Other demos

These demos are duplicated with mmedting_inference_demo.py and may be removed in the future.

- [colorization_demo.py](./colorization_demo.py)
- [conditional_demo.py](./conditional_demo.py)
- [inpainting_demo.py](./inpainting_demo.py)
- [matting_demo.py](./matting_demo.py)
- [restoration_demo.py](./restoration_demo.py)
- [restoration_video_demo.py](./restoration_video_demo.py)
- [translation_demo.py](./translation_demo.py)
- [unconditional_demo.py](./unconditional_demo.py)
- [video_interpolation_demo.py](./video_interpolation_demo.py)
