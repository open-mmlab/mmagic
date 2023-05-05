# NPU (HUAWEI Ascend)

## Usage

Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV and [mmengine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html#build-from-source) on NPU devices.

Here we use 8 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/edsr/edsr_x2c64b16_1xb16-300k_div2k.py 8
```

Also, you can use only one NPU to train the model with the following command:

```shell
python tools/train.py configs/edsr/edsr_x2c64b16_1xb16-300k_div2k.py
```

## Models Results

|                                           Model                                            | Dataset | PSNR  | SSIM | Download                                                                                       |
| :----------------------------------------------------------------------------------------: | ------- | :---: | :--- | :--------------------------------------------------------------------------------------------- |
| [edsr_x2c64b16_1x16_300k_div2k](https://github.com/open-mmlab/mmagic/blob/main/configs/edsr/edsr_x2c64b16_1xb16-300k_div2k.py) | DIV2K   | 35.83 | 0.94 | [log](https://download.openmmlab.com/mmediting/device/npu/edsr/edsr_x2c64b16_1xb16-300k_div2k.log) |

**Notes:**

- If not specially marked, the results on NPU with amp are the basically same as those on the GPU with FP32.

**All above models are provided by Huawei Ascend group.**
