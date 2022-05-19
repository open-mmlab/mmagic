### 演示

我们针对特定任务提供了一些脚本，可以对单张图像进行推理。

#### 图像补全

您可以使用以下命令，输入一张测试图像以及缺损部位的遮罩图像，实现对测试图像的补全。

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

如果指定了 --imshow ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/inpainting_demo.py \
    configs/inpainting/global_local/gl_256x256_8x12_celeba.py \
    https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth \
    tests/data/image/celeba_test.png \
    tests/data/image/bbox_mask.png \
    tests/data/pred/inpainting_celeba.png
```

补全结果将保存在 `tests/data/pred/inpainting_celeba.png` 中。

#### 抠图

您可以使用以下命令，输入一张测试图像以及对应的三元图（trimap），实现对测试图像的抠图。

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

如果指定了 --imshow ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/matting_demo.py \
    configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py \
    work_dirs/dim_stage3/latest.pth \
    tests/data/merged/GT05.jpg \
    tests/data/trimap/GT05.png \
    tests/data/pred/GT05.png
```

预测的 alpha 遮罩将保存在 `tests/data/pred/GT05.png` 中。

#### 图像超分辨率

您可以使用以下命令来测试要恢复的图像。

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

如果指定了 `--imshow` ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/restoration_demo.py \
    configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py \
    work_dirs/esrgan_x4c64b23g32_g1_400k_div2k/latest.pth \
    tests/data/lq/baboon_x4.png \
    demo/demo_out_baboon.png
```

您可以通过提供 `--ref-path` 参数来测试基于参考的超分辨率算法。例子：

```shell
python demo/restoration_demo.py \
    configs/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED.py \
    https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth \
    tests/data/test_multiple_gt/sequence_1/00000000.png \
    work_dirs/demo_out.png \
    --ref-path tests/data/test_multiple_gt/sequence_1/00000001.png
```

#### 人脸图像超分辨率

您可以使用以下命令来测试要恢复的人脸图像。

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

如果指定了 --imshow ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/restoration_face_demo.py \
    configs/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq.py \
    https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth \
    tests/data/face/000001.png \
    results/000001.png \
    --upscale-factor 4
```

#### 视频超分辨率

您可以使用以下命令来测试视频以进行恢复。

```shell
python demo/restoration_video_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${INPUT_DIR} \
    ${OUTPUT_DIR} \
    [--window-size=${WINDOW_SIZE}] \
    [--device ${GPU_ID}]
```

它同时支持滑动窗口框架和循环框架。 例子：

EDVR:

```shell
python demo/restoration_video_demo.py \
    ./configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py \
    https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth \
    data/Vid4/BIx4/calendar/ \
    ./output \
    --window-size=5
```

BasicVSR:

```shell
python demo/restoration_video_demo.py \
    ./configs/restorers/basicvsr/basicvsr_reds4.py \
    https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth \
    data/Vid4/BIx4/calendar/ \
    ./output
```

复原的视频将保存在 `output/` 中。

#### 视频插帧

您可以使用以下命令来测试视频插帧。

```shell
python demo/video_interpolation_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${INPUT_DIR} \
    ${OUTPUT_DIR} \
    [--fps-multiplier ${FPS_MULTIPLIER}] \
    [--fps ${FPS}]
```

`${INPUT_DIR}` 和 `${OUTPUT_DIR}` 可以是视频文件路径或存放一系列有序图像的文件夹。
若 `${OUTPUT_DIR}` 是视频文件地址，其帧率可由输入视频帧率和 `fps_multiplier` 共同决定，也可由 `fps` 直接给定(其中前者优先级更高）。例子：

由输入视频帧率和 `fps_multiplier` 共同决定输出视频的帧率：

```shell
python demo/video_interpolation_demo.py \
    configs/video_interpolators/cain/cain_b5_320k_vimeo-triplet.py \
    https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_320k_vimeo-triple_20220117-647f3de2.pth \
    tests/data/test_inference.mp4 \
    tests/data/test_inference_vfi_out.mp4 \
    --fps-multiplier 2.0
```

由 `fps` 直接给定输出视频的帧率：

```shell
python demo/video_interpolation_demo.py \
    configs/video_interpolators/cain/cain_b5_320k_vimeo-triplet.py \
    https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_320k_vimeo-triple_20220117-647f3de2.pth \
    tests/data/test_inference.mp4 \
    tests/data/test_inference_vfi_out.mp4 \
    --fps 60.0
```

#### 图像生成

```shell
python demo/generation_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${IMAGE_FILE} \
    ${SAVE_FILE} \
    [--unpaired-path ${UNPAIRED_IMAGE_FILE}] \
    [--imshow] \
    [--device ${GPU_ID}]
```

如果指定了 `--unpaired-path` （用于 CycleGAN），模型将执行未配对的图像到图像的转换。 如果指定了 `--imshow` ，演示也将使用opencv显示图像。 例子：

针对配对数据：

```shell
python demo/generation_demo.py \
    configs/example_config.py \
    work_dirs/example_exp/example_model_20200202.pth \
    demo/demo.jpg \
    demo/demo_out.jpg
```

针对未配对数据（用 opencv 显示图像）：

```shell
python demo/generation_demo.py 、
    configs/example_config.py \
    work_dirs/example_exp/example_model_20200202.pth \
    demo/demo.jpg \
    demo/demo_out.jpg \
    --unpaired-path demo/demo_unpaired.jpg \
    --imshow
```
