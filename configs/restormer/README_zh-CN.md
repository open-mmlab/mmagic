# Restormer (CVPR 2022 -- Oral)

> [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)

> **任务**: 图像复原

<!-- [ALGORITHM] -->

## Results and models

使用 PSNR 和 SSIM 作为指标，对所有任务进行评测

所有恢复任务的模型权重链接：https://pan.baidu.com/s/1TWWR_x0gRvfZoK7Fux6prQ?pwd=o6un(o6un).

<details>
<summary>图像去雨</summary>

|          |  PSNR   |  SSIM  |                                                                配置文件                                                                |
| :------: | :-----: | :----: | :------------------------------------------------------------------------------------------------------------------------------------: |
| Test100  | 32.0291 | 0.9239 |  [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py)  |
| Rain100H | 31.4800 | 0.9056 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100H](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100H.py) |
| Rain100L | 39.1022 | 0.9787 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100L](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100L.py) |
| Test2800 | 34.2170 | 0.9451 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test2800](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test2800.py) |
| Test1200 | 33.2253 | 0.9272 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test1200](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test1200.py) |
| Average  | 34.0107 | 0.9361 |                                                                   -                                                                    |

</details>

<details>
<summary>图像去模糊</summary>

|            |  PSNR   |  SSIM  |                                                                 配置文件                                                                  |
| :--------: | :-----: | :----: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|   GoPro    | 32.9295 | 0.9404 |     [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py)     |
|    HIDE    | 31.2289 | 0.9231 |      [restormer_d48nb4668nrb4h1248-lr3e-4-300k_HIDE](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_HIDE.py)      |
| RealBlur-R | 35.9141 | 0.9707 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurR](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurR.py) |
| RealBlur-J | 28.4356 | 0.8681 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurR](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurJ.py) |

</details>

<details>
<summary>图像去噪</summary>

**灰度图的高斯噪声**

| Type                      | Set12        | PSNR    | SSIM   | 配置文件                                                                                                                                               |
| ------------------------- | ------------ | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each noise level Training | $\\sigma$=15 | 33.4659 | 0.9137 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma15.py) |
|                           | $\\sigma$=25 | 31.0880 | 0.8397 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma25.py) |
|                           | $\\sigma$=50 | 28.0214 | 0.8127 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma50.py) |
| Blind                     | $\\sigma$=15 | 33.4035 | 0.9125 |                                                                                                                                                        |
|                           | $\\sigma$=25 | 31.0880 | 0.8770 |                                                                                                                                                        |
|                           | $\\sigma$=50 | 28.0145 | 0.8125 |                                                                                                                                                        |

| Type                      | BSD68        | PSNR    | SSIM   | 配置文件                                                                                                                                               |
| ------------------------- | ------------ | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each noise level Training | $\\sigma$=15 | 31.9535 | 0.8974 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py) |
|                           | $\\sigma$=25 | 29.5355 | 0.8397 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma25.py) |
|                           | $\\sigma$=50 | 26.6082 | 0.7422 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma50.py) |
| Blind                     | $\\sigma$=15 | 30.6039 | 0.8385 |                                                                                                                                                        |
|                           | $\\sigma$=25 | 28.0272 | 0.7525 |                                                                                                                                                        |
|                           | $\\sigma$=50 | 25.0556 | 0.6174 |                                                                                                                                                        |

| Type                      | Urban100     | PSNR    | SSIM   | 配置文件                                                                                                                                               |
| ------------------------- | ------------ | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each noise level Training | $\\sigma$=15 | 34.4359 | 0.9420 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma15.py) |
|                           | $\\sigma$=25 | 32.1171 | 0.9141 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma25.py) |
|                           | $\\sigma$=50 | 28.9632 | 0.8570 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma50.py) |
| Blind                     | $\\sigma$=15 | 34.3165 | 0.9410 |                                                                                                                                                        |
|                           | $\\sigma$=25 | 32.0490 | 0.9132 |                                                                                                                                                        |
|                           | $\\sigma$=50 | 28.9847 | 0.8571 |                                                                                                                                                        |

**彩色图像的高斯噪声**

| Type                      | CBSD68       | PSNR    | SSIM   | 配置文件                                                                                                                                               |
| ------------------------- | ------------ | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each noise level Training | $\\sigma$=15 | 34.3513 | 0.9352 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py) |
|                           | $\\sigma$=25 | 31.7534 | 0.8945 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma25.py) |
|                           | $\\sigma$=50 | 28.5511 | 0.8125 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma50.py) |
| Blind                     | $\\sigma$=15 | 34.3404 | 0.9356 |                                                                                                                                                        |
|                           | $\\sigma$=25 | 31.7414 | 0.8946 |                                                                                                                                                        |
|                           | $\\sigma$=50 | 28.5506 | 0.8127 |                                                                                                                                                        |

| Type                      | Kodak24      | PSNR    | SSIM   | 配置文件                                                                                                                                               |
| ------------------------- | ------------ | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each noise level Training | $\\sigma$=15 | 35.4924 | 0.9312 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma15.py) |
|                           | $\\sigma$=25 | 33.0608 | 0.8943 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma25.py) |
|                           | $\\sigma$=50 | 30.0220 | 0.8241 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma50.py) |
| Blind                     | $\\sigma$=15 | 35.4573 | 0.9308 |                                                                                                                                                        |
|                           | $\\sigma$=25 | 33.0368 | 0.8939 |                                                                                                                                                        |
|                           | $\\sigma$=50 | 30.0118 | 0.8238 |                                                                                                                                                        |

| Type                      | McMaster     | PSNR    | SSIM   | 配置文件                                                                                                                                               |
| ------------------------- | ------------ | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each noise level Training | $\\sigma$=15 | 35.6177 | 0.9353 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma15.py) |
|                           | $\\sigma$=25 | 33.3405 | 0.9069 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma25.py) |
|                           | $\\sigma$=50 | 30.2738 | 0.8521 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma50.py) |
| Blind                     | $\\sigma$=15 | 35.5634 | 0.9345 |                                                                                                                                                        |
|                           | $\\sigma$=25 | 33.3111 | 0.9064 |                                                                                                                                                        |
|                           | $\\sigma$=50 | 30.2751 | 0.8522 |                                                                                                                                                        |

| Type                      | Urban100     | PSNR    | SSIM   | 配置文件                                                                                                                                               |
| ------------------------- | ------------ | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each noise level Training | $\\sigma$=15 | 35.1535 | 0.9530 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma15.py) |
|                           | $\\sigma$=25 | 32.9670 | 0.9317 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma25.py) |
|                           | $\\sigma$=50 | 30.0252 | 0.8902 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma50.py) |
| Blind                     | $\\sigma$=15 | 35.0756 | 0.9524 |                                                                                                                                                        |
|                           | $\\sigma$=25 | 32.9174 | 0.9312 |                                                                                                                                                        |
|                           | $\\sigma$=50 | 30.0189 | 0.8897 |                                                                                                                                                        |

**真实场景图像去噪**

|      | PSNR    | SSIM   | MAE    | Config                                                                                                                                             |
| ---- | ------- | ------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| SIDD | 40.0156 | 0.9603 | 0.0086 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD](/configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py) |
| DND  | -       | -      | -      | -                                                                                                                                                  |

</details>

<details>
<summary>图像去失焦模糊</summary>

|        | PSNR    | SSIM   | MAE    | Config                                                                                                                                             |
| ------ | ------- | ------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| DPDD_S | 25.9805 | 0.8166 | 0.0378 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD](/configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py) |
| DPDD_D | 26.6160 | 0.8346 | 0.0354 | [restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD](/configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py)     |

</details>

## 使用方法

**训练命令**

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
# Motion Deblurring
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py

# Deraining
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py

# Defocus Deblurring
#Single Image
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py
#Dual Image
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py

# Denoising
# Color Gaussian Noise
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py
# Grayscale Gaussian Noise
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py

# single-gpu train
# Motion Deblurring
python tools/train.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py

# Deraining
python tools/train.py configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py

# Defocus Deblurring
#Single Image
python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py
#Dual Image
python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py

# Denoising
# Color Gaussian Noise
python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py
# Grayscale Gaussian Noise
python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py

# multi-gpu train
# Motion Deblurring
./tools/dist_train.sh configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py 8

# Deraining
./tools/dist_train.sh configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py 8

# Defocus Deblurring
#Single Image
./tools/dist_train.sh configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py 8
#Dual Image
./tools/dist_train.sh configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py 8

# Denoising
# Color Gaussian Noise
./tools/dist_train.sh configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py 8
# Grayscale Gaussian Noise
./tools/dist_train.sh configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py 8

```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

**测试命令**

```shell
# cpu test
# Motion Deblurring
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py Motion_Deblurring_checkpoint_path

# single-gpu test
# Motion Deblurring
python tools/test.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py Motion_Deblurring_checkpoint_path

# multi-gpu test
# Motion Deblurring
./tools/dist_test.sh configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py Motion_Deblurring_checkpoint_path


```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

```bibtex
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```
