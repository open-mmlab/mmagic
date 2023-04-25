# NPU (华为昇腾)

## 使用方法

首先，请参考[MMCV](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#npu-mmcv-full) 安装带有 NPU 支持的 MMCV与 [mmengine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html#build-from-source) 。

使用如下命令，可以利用 8 个 NPU 训练模型（以 edsr 为例）：

```shell
bash tools/dist_train.sh configs/edsr/edsr_x2c64b16_1xb16-300k_div2k.py 8
```

或者，使用如下命令，在一个 NPU 上训练模型（以 edsr 为例）：

```shell
python tools/train.py configs/edsr/edsr_x2c64b16_1xb16-300k_div2k.py
```

## 经过验证的模型

|                                           Model                                            | Dataset | PSNR  | SSIM | Download                                                                                       |
| :----------------------------------------------------------------------------------------: | ------- | :---: | :--- | :--------------------------------------------------------------------------------------------- |
| [edsr_x2c64b16_1x16_300k_div2k](https://github.com/open-mmlab/mmagic/blob/main/configs/edsr/edsr_x2c64b16_1xb16-300k_div2k.py) | DIV2K   | 35.83 | 0.94 | [log](https://download.openmmlab.com/mmediting/device/npu/edsr/edsr_x2c64b16_1xb16-300k_div2k.log) |

**注意:**

- 如果没有特别标记，NPU 上的结果与使用 FP32 的 GPU 上的结果相同。

**以上所有模型权重及训练日志均由华为昇腾团队提供**
