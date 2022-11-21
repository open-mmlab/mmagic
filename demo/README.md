# MMEditing Demo

There are some mmediting demos for you to run with command line in this folder.

We provide python command line usage here to run these demos and mode instructions could also be found on the [web page](https://mmediting.readthedocs.io/en/dev-1.x/user_guides/3_inference.html)

## Download sample images or videos

We prepared some images and videos for you to run demo with. After MMEdit is well installed, you could use demo in this folder to infer these data. Download at here (url) and extract it to MMEdit root path.

```shell
cd mmediting
wget url
unzip resources.zip
```

## MMEditing inference demo

You can use the following commands to perform inference with a MMEdit model.

Usage of python API can be found in this [tutotial](demo/mmediting_inference_tutorial.ipynb).

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

1. Conditional GANs

```shell
python demo/mmediting_inference_demo.py \
        --model-name biggan \
        --label 1 \
        --result-out-dir resources/output/conditional/demo_conditional_biggan_res.jpg \
```

2. Inpainting

```shell
python demo/mmediting_inference_demo.py \
        --model-name global_local  \
        --img resources/input/inpainting/celeba_test.png \
        --mask resources/input/inpainting/bbox_mask.png \
        --result-out-dir resources/output/inpainting/demo_inpainting_global_local_res.jpg
```

3. Matting

```shell
python demo/mmediting_inference_demo.py \
        --model-name global_local  \
        --img resources/input/matting/GT05.jpg \
        --mask resources/input/matting/GT05_trimap.jpg \
        --result-out-dir resources/output/matting/demo_matting_gca_res.png
```

4. Super resolution

```shell
python demo/mmediting_inference_demo.py \
        --model-name esrgan \
        --img resources/input/restoration/0901x2.png \
        --result-out-dir resources/output/restoration/demo_restoration_esrgan_res.png
```

5. Image translation

```shell
python demo/mmediting_inference_demo.py \
        --model-name pix2pix \
        --img resources/input/translation/gt_mask_0.png \
        --result-out-dir resources/output/translation/demo_translation_pix2pix_res.png
```

6. Unconditional GANs

```shell
python demo/mmediting_inference_demo.py \
        --model-name styleganv1 \
        --result-out-dir resources/output/unconditional/demo_unconditional_styleganv1_res.jpg
```

7. Video interpolation

```shell
python demo/mmediting_inference_demo.py \
        --model-name flavr \
        --video resources/input/video_interpolation/b-3LLDhc4EU_000000_000010.mp4 \
        --result-out-dir resources/output/video_interpolation/demo_video_interpolation_flavr_res.mp4
```

8. Video restoration

```shell
python demo/mmediting_inference_demo.py \
        --model-name edvr \
        --extra-parameters window_size=5 \
        --video resources/input/video_interpolation/QUuC4vJs_000084_000094_400x320.mp4 \
        --result-out-dir resources/output/video_restoration/demo_video_restoration_edvr_res.mp4
```

## Face restoration demo

You can use the following commands to test an face image for restoration.

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

Examples:

```shell
python demo/restoration_face_demo.py \
    configs/glean/glean_in128out1024_4x2_300k_ffhq_celebahq.py \
    https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth \
    tests/data/image/face/000001.png \
    tests/data/pred/000001.png \
    --upscale-factor 4
```

## Other demos

These demos are duplicated with mmedting_inference_demo.py and may be removed in the future.

- colorization_demo.py
- conditional_demo.py
- inpainting_demo.py
- matting_demo.py
- restoration_demo.py
- restoration_video_demo.py
- translation_demo.py
- unconditional_demo.py
- video_interpolation_demo.py
