### 演示

我们针对特定任务提供了一些脚本，可以对单张图像进行推理。

#### 图像补全

您可以使用以下命令，输入一张测试图像以及缺损部位的遮罩图像，实现对测试图像的补全。

```shell
python demo/inpainting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MASKED_IMAGE_FILE} ${MASK_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果指定了 --imshow ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/inpainting_demo.py configs/inpainting/global_local/gl_256x256_8x12_celeba.py work_dirs/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth tests/data/image/celeba_test.png tests/data/image/bbox_mask.png tests/data/pred/inpainting_celeba.png
```

补全结果将保存在 `tests/data/pred/inpainting_celeba.png` 中。

#### 抠图

您可以使用以下命令，输入一张测试图像以及对应的三元图（trimap），实现对测试图像的抠图。

```shell
python demo/matting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${TRIMAP_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果指定了 --imshow ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/matting_demo.py configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py work_dirs/dim_stage3/latest.pth tests/data/merged/GT05.jpg tests/data/trimap/GT05.png tests/data/pred/GT05.png
```

预测的 alpha 遮罩将保存在 `tests/data/pred/GT05.png` 中。

#### 图像超分辨率

您可以使用以下命令来测试要恢复的图像。

```shell
python demo/restoration_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果指定了 --imshow ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/restoration_demo.py configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py work_dirs/esrgan_x4c64b23g32_g1_400k_div2k/latest.pth tests/data/lq/baboon_x4.png demo/demo_out_baboon.png
```

#### 人脸图像超分辨率

您可以使用以下命令来测试要恢复的人脸图像。

```shell
python demo/restoration_face_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--upscale_factor] [--face_size] [--imshow] [--device ${GPU_ID}]
```

如果指定了 --imshow ，演示程序将使用 opencv 显示图像。例子：

```shell
python demo/restoration_face_demo.py configs/restorers/glean/glean_in128out1024_2x4_300k_ffhq_celebahq.py https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth tests/data/face/000001.png results/000001.png --upscale_factor 4
```

#### 视频超分辨率

您可以使用以下命令来测试视频以进行恢复。

```shell
python demo/restoration_video_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${INPUT_DIR} ${OUTPUT_DIR} [--window_size=$WINDOW_SIZE] [--device ${GPU_ID}]
```

它同时支持滑动窗口框架和循环框架。 例子：

EDVR:

```shell
python demo/restoration_video_demo.py ./configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth data/Vid4/BIx4/calendar/ ./output --window_size=5
```

BasicVSR:

```shell
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth data/Vid4/BIx4/calendar/ ./output
```

复原的视频将保存在 `output/` 中。

#### 图像生成

```shell
python demo/generation_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--unpaired_path ${UNPAIRED_IMAGE_FILE}] [--imshow] [--device ${GPU_ID}]
```

如果指定了 `--unpaired_path` （用于 CycleGAN），模型将执行未配对的图像到图像的转换。 如果指定了 `--imshow` ，演示也将使用opencv显示图像。 例子：

针对配对数据：

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg
```

针对未配对数据（用 opencv 显示图像）：

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg --unpaired_path demo/demo_unpaired.jpg --imshow
```
