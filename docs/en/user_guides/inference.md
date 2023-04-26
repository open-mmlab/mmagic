# Tutorial 3: Inference with pre-trained models

MMagic provides APIs for you to easily play with state-of-the-art models on your own images or videos.
Specifically, MMagic supports various fundamental generative models, including:
unconditional Generative Adversarial Networks (GANs), conditional GANs, internal learning, diffusion models, etc.
MMagic also supports various applications, including:
image super-resolution, video super-resolution, video frame interpolation, image inpainting, image matting, image-to-image translation, etc.

In this section, we will specify how to play with our pre-trained models.

- [Tutorial 3: Inference with pre-trained models](#tutorial-3-inference-with-pre-trained-models)
  - [Sample images with unconditional GANs](#sample-images-with-unconditional-gans)
  - [Sample images with conditional GANs](#sample-images-with-conditional-gans)
  - [Sample images with diffusion models](#sample-images-with-diffusion-models)
  - [Run a demo of image inpainting](#run-a-demo-of-image-inpainting)
  - [Run a demo of image matting](#run-a-demo-of-image-matting)
  - [Run a demo of image super-resolution](#run-a-demo-of-image-super-resolution)
  - [Run a demo of facial restoration](#run-a-demo-of-facial-restoration)
  - [Run a demo of video super-resolution](#run-a-demo-of-video-super-resolution)
  - [Run a demo of video frame interpolation](#run-a-demo-of-video-frame-interpolation)
  - [Run a demo of image translation models](#run-a-demo-of-image-translation-models)

## Sample images with unconditional GANs

MMagic provides high-level APIs for sampling images with unconditional GANs. Here is an example of building StyleGAN2-256 and obtaining the synthesized images.

```python
from mmagic.apis import init_model, sample_unconditional_model

# Specify the path to model config and checkpoint file
config_file = 'configs/styleganv2/stylegan2_c2_8xb4_ffhq-1024x1024.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth'

device = 'cuda:0'
# init a generative model
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_unconditional_model(model, 4)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/unconditional_demo.py](../../../demo/unconditional_demo.py) with the following commands:

```shell
python demo/unconditional_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

Note that more arguments are also offered to customize your sampling procedure. Please use `python demo/unconditional_demo.py --help` to check more details.

## Sample images with conditional GANs

MMagic provides high-level APIs for sampling images with conditional GANs. Here is an example for building SAGAN-128 and obtaining the synthesized images.

```python
from mmagic.apis import init_model, sample_conditional_model

# Specify the path to model config and checkpoint file
config_file = 'configs/sagan/sagan_woReLUinplace-Glr1e-4_Dlr4e-4_noaug-ndisc1-8xb32-bigGAN-sch_imagenet1k-128x128.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmediting/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth'

device = 'cuda:0'
# init a generative model
model = init_model(config_file, checkpoint_file, device=device)
# sample images with random label
fake_imgs = sample_conditional_model(model, 4)

# sample images with the same label
fake_imgs = sample_conditional_model(model, 4, label=0)

# sample images with specific labels
fake_imgs = sample_conditional_model(model, 4, label=[0, 1, 2, 3])
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/conditional_demo.py](../../../demo/conditional_demo.py) with the following commands:

```shell
python demo/conditional_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--label] ${LABEL} \
    [--samples-per-classes] ${SAMPLES_PER_CLASSES} \
    [--sample-all-classes] \
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

If `--label` is not passed, images with random labels would be generated.
If `--label` is passed, we would generate `${SAMPLES_PER_CLASSES}` images for each input label.
If `sample_all_classes` is set true in command line, `--label` would be ignored and the generator will output images for all categories.

Note that more arguments are also offered to customizing your sampling procedure. Please use `python demo/conditional_demo.py --help` to check more details.

## Sample images with diffusion models

MMagic provides high-level APIs for sampling images with diffusion models. Here is an example for building I-DDPM and obtaining the synthesized images.

```python
from mmagic.apis import init_model, sample_ddpm_model

# Specify the path to model config and checkpoint file
config_file = 'configs/improved_ddpm/ddpm_cosine-hybird-timestep-4k_16xb8-1500kiters_imagenet1k-64x64.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmediting/improved_ddpm/ddpm_cosine_hybird_timestep-4k_imagenet1k_64x64_b8x16_1500k_20220103_223919-b8f1a310.pth'
device = 'cuda:0'
# init a generative model
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_ddpm_model(model, 4)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/ddpm_demo.py](https://github.com/open-mmlab/mmagic/blob/main/demo/ddpm_demo.py) with the following commands:

```shell
python demo/ddpm_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

Note that more arguments are also offered to customizing your sampling procedure. Please use `python demo/ddpm_demo.py --help` to check more details.

## Run a demo of image inpainting

You can use the following commands to test images for inpainting.

```shell
python demo/inpainting_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${MASKED_IMAGE_FILE} \
    ${MASK_FILE} \
    ${SAVE_FILE} \
    [--imshow] \
    [--device ${GPU_ID}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/inpainting_demo.py \
    configs/global_local/gl_256x256_8x12_celeba.py \
    https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth \
    tests/data/image/celeba_test.png \
    tests/data/image/bbox_mask.png \
    tests/data/pred/inpainting_celeba.png
```

The predicted inpainting result will be saved in `tests/data/pred/inpainting_celeba.png`.

## Run a demo of image matting

You can use the following commands to test a pair of images and trimap.

```shell
python demo/matting_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${IMAGE_FILE} \
    ${TRIMAP_FILE} \
    ${SAVE_FILE} \
    [--imshow] \
    [--device ${GPU_ID}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/matting_demo.py \
    configs/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py \
    https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth \
    tests/data/matting_dataset/merged/GT05.jpg \
    tests/data/matting_dataset/trimap/GT05.png \
    tests/data/pred/GT05.png
```

The predicted alpha matte will be saved in `tests/data/pred/GT05.png`.

## Run a demo of image super-resolution

You can use the following commands to test an image for restoration.

```shell
python demo/restoration_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${IMAGE_FILE} \
    ${SAVE_FILE} \
    [--imshow] \
    [--device ${GPU_ID}] \
    [--ref-path ${REF_PATH}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/restoration_demo.py \
    configs/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py \
    https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth \
    tests/data/image/lq/baboon_x4.png \
    demo/demo_out_baboon.png
```

You can test Ref-SR by providing `--ref-path`. Examples:

```shell
python demo/restoration_demo.py \
    configs/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED.py \
    https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth \
    tests/data/frames/sequence/gt/sequence_1/00000000.png \
    demo/demo_out.png \
    --ref-path tests/data/frames/sequence/gt/sequence_1/00000001.png
```

## Run a demo of facial restoration

You can use the following commands to test a face image for restoration.

```shell
python demo/restoration_face_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${IMAGE_FILE} \
    ${SAVE_FILE} \
    [--upscale-factor] \
    [--face-size] \
    [--imshow] \
    [--device ${GPU_ID}]
```

If `--imshow` is specified, the demo will also show image with opencv. Examples:

```shell
python demo/restoration_face_demo.py \
    configs/glean/glean_in128out1024_4xb2-300k_ffhq-celeba-hq.py \
    https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth \
    tests/data/image/face/000001.png \
    tests/data/pred/000001.png \
    --upscale-factor 4
```

## Run a demo of video super-resolution

You can use the following commands to test a video for restoration.

```shell
python demo/restoration_video_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${INPUT_DIR} \
    ${OUTPUT_DIR} \
    [--window-size=${WINDOW_SIZE}] \
    [--device ${GPU_ID}]
```

It supports both the sliding-window framework and the recurrent framework. Examples:

EDVR:

```shell
python demo/restoration_video_demo.py \
    configs/edvr/edvrm_wotsa_x4_g8_600k_reds.py \
    https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth \
    data/Vid4/BIx4/calendar/ \
    demo/output \
    --window-size=5
```

BasicVSR:

```shell
python demo/restoration_video_demo.py \
    configs/basicvsr/basicvsr_reds4.py \
    https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth \
    data/Vid4/BIx4/calendar/ \
    demo/output
```

The restored video will be saved in `output/`.

## Run a demo of video frame interpolation

You can use the following commands to test a video for frame interpolation.

```shell
python demo/video_interpolation_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${INPUT_DIR} \
    ${OUTPUT_DIR} \
    [--fps-multiplier ${FPS_MULTIPLIER}] \
    [--fps ${FPS}]
```

`${INPUT_DIR}` / `${OUTPUT_DIR}` can be a path of video file or the folder of a sequence of ordered images.
If `${OUTPUT_DIR}` is a path of video file, its frame rate can be determined by the frame rate of input video and `fps_multiplier`, or be determined by `fps` directly (the former has higher priority). Examples:

The frame rate of output video is determined by the frame rate of input video and `fps_multiplier`：

```shell
python demo/video_interpolation_demo.py \
    configs/cain/cain_b5_g1b32_vimeo90k_triplet.py \
    https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_320k_vimeo-triple_20220117-647f3de2.pth \
    tests/data/test_inference.mp4 \
    tests/data/test_inference_vfi_out.mp4 \
    --fps-multiplier 2.0
```

The frame rate of output video is determined by `fps`:

```shell
python demo/video_interpolation_demo.py \
    configs/cain/cain_b5_g1b32_vimeo90k_triplet.py \
    https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_320k_vimeo-triple_20220117-647f3de2.pth \
    tests/data/test_inference.mp4 \
    tests/data/test_inference_vfi_out.mp4 \
    --fps 60.0
```

## Run a demo of image translation models

MMagic provides high-level APIs for translating images by using image translation models. Here is an example of building Pix2Pix and obtaining the translated images.

```python
from mmagic.apis import init_model, sample_img2img_model

# Specify the path to model config and checkpoint file
config_file = 'configs/pix2pix/pix2pix_vanilla-unet-bn_wo-jitter-flip-4xb1-190kiters_edges2shoes.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmediting/pix2pix/refactor/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_convert-bgr_20210902_170902-0c828552.pth'
# Specify the path to image you want to translate
image_path = 'tests/data/paired/test/33_AB.jpg'
device = 'cuda:0'
# init a generative model
model = init_model(config_file, checkpoint_file, device=device)
# translate a single image
translated_image = sample_img2img_model(model, image_path, target_domain='photo')
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/translation_demo.py](../../../demo/translation_demo.py) with the following commands:

```shell
python demo/translation_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    ${IMAGE_PATH}
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

Note that more customized arguments are also offered to customize your sampling procedure. Please use `python demo/translation_demo.py --help` to check more details.
