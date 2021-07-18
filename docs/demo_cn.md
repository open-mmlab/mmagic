### 演示

我们提供了一些用于特定任务的演示脚本来测试单张图像。、

### 修复

你可以使用以下命令来测试一对图像进行修复。

```shell
python demo/inpainting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MASKED_IMAGE_FILE} ${MASK_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果指定 `--imshow`  参数，演示也会通过opencv显示图像。例如：

```shell
python demo/inpainting_demo.py configs/inpainting/global_local/gl_256x256_8x12_celeba.py work_dirs/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth tests/data/image/celeba_test.png tests/data/image/bbox_mask.png tests/data/pred/inpainting_celeba.png
```

预测的修复结果将会保存在 `tests/data/pred/inpainting_celeba.png`目录下。

### 抠图

你可以使用如下命令来测试一对图像和三元图进行抠图。

```shell
python demo/matting_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${TRIMAP_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果指定 `--imshow`  参数，演示也会通过opencv显示图像。例如：

```shell
python demo/matting_demo.py configs/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py work_dirs/dim_stage3/latest.pth tests/data/merged/GT05.jpg tests/data/trimap/GT05.png tests/data/pred/GT05.png
```

预测的alpha遮罩图会保存在`tests/data/pred/GT05.png`目录下。

### 超分辨率重建（图像）

你可以使用如下命令来测试一张图像用于超分辨率重建。

```shell
python demo/restoration_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--imshow] [--device ${GPU_ID}]
```

如果指定 `--imshow`  参数，演示也会通过opencv显示图像。例如：

```shell
python demo/restoration_demo.py configs/restorer/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k.py work_dirs/esrgan_x4c64b23g32_1x16_400k_div2k/latest.pth tests/data/lq/baboon_x4.png demo/demo_out_baboon.png
```

### 超分辨率重建（视频）

你可以使用如下命令来测试一段视频序列用于超分辨率重建。

```shell
python demo/restoration_video_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${INPUT_DIR} ${OUTPUT_DIR} [--window_size=$WINDOW_SIZE] [--device ${GPU_ID}]
```

本项目同时支持基于滑动窗口框架和基于循环网络框架的模型。例如：

EDVR：

```shell
python demo/restoration_video_demo.py ./configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth data/Vid4/BIx4/calendar/ ./output --window_size=5
```

BasicVSR：

```shell
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth data/Vid4/BIx4/calendar/ ./output
```

重建之后的视频将会被保存在`output/`目录下。

### 生成

```shell
python demo/generation_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${IMAGE_FILE} ${SAVE_FILE} [--unpaired_path ${UNPAIRED_IMAGE_FILE}] [--imshow] [--device ${GPU_ID}]
```

如果指定了`--unpaired_path` 参数（用于 CycleGAN 模型），模型将会执行不匹配的图像-图像转换。如果指定了`--imshow` 参数，演示程序会通过opencv显示图像。例如：

匹配的：

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg
```

不匹配的（也通过opencv显示图像）：

```shell
python demo/generation_demo.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth demo/demo.jpg demo/demo_out.jpg --unpaired_path demo/demo_unpaired.jpg --imshow
```

