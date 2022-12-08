# Restormer (CVPR 2022 -- Oral)

> [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)

> **Task**: Image Restoration

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).

<!-- [IMAGE] -->

<div align=center >
 <img src="https://camo.githubusercontent.com/3c863280a592f3535e4c2411946db0e3495b567fe400bd97f5a8b46f2d3cdcef/68747470733a2f2f692e696d6775722e636f6d2f756c4c6f4569672e706e67" width="800"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

The model weights of different image restoration tasks are downloaded [here(o6un)](https://pan.baidu.com/s/1TWWR_x0gRvfZoK7Fux6prQ?pwd=o6un).

### **Deraining**

| Method                                                                         | Dataset  | PSNR    | SSIM   | GPU Info | Download |
| ------------------------------------------------------------------------------ | -------- | ------- | ------ | -------- | -------- |
| [restormer_official_rain13k](/configs/restormer/restormer_official_rain13k.py) | Test100  | 32.0291 | 0.9239 | 1        | -        |
| [restormer_official_rain13k](/configs/restormer/restormer_official_rain13k.py) | Rain100H | 31.4800 | 0.9056 | 1        | -        |
| [restormer_official_rain13k](/configs/restormer/restormer_official_rain13k.py) | Rain100L | 39.1022 | 0.9787 | 1        | -        |
| [restormer_official_rain13k](/configs/restormer/restormer_official_rain13k.py) | Test2800 | 34.2170 | 0.9451 | 1        | -        |
| [restormer_official_rain13k](/configs/restormer/restormer_official_rain13k.py) | Test1200 | 33.2253 | 0.9272 | 1        | -        |
| [restormer_official_rain13k](/configs/restormer/restormer_official_rain13k.py) | Average  | 34.0107 | 0.9361 | 1        | -        |

### **Deblurring**

| Method                                                                       | Datasets   | PSNR    | SSIM   | GPU Info | Download |
| ---------------------------------------------------------------------------- | ---------- | ------- | ------ | -------- | -------- |
| [restormer_official_deblur](/configs/restormer/restormer_official_deblur.py) | GoPro      | 32.9295 | 0.9496 | 1        | -        |
| [restormer_official_deblur](/configs/restormer/restormer_official_deblur.py) | HIDE       | 31.2289 | 0.9345 | 1        | -        |
| [restormer_official_deblur](/configs/restormer/restormer_official_deblur.py) | RealBlur-J | 28.4356 | 0.8681 | 1        | -        |
| [restormer_official_deblur](/configs/restormer/restormer_official_deblur.py) | RealBlur-R | 35.9141 | 0.9707 | 1        | -        |

### **Denoising**

**Test Grayscale Gaussian Noise: $\\sigma$=15**

| Method                                                                                                                         | Dataset  | PSNR    | SSIM   | GPU Info | ckpt                                | Download |
| ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------- | ------ | -------- | ----------------------------------- | -------- |
| [restormer_official_gaussian_denoising_gray_sigma15](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma15.py) | Set12    | 34.0182 | 0.9160 | 1        | gaussian_gray_denoising_sigma15.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma15](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma15.py) | BSD68    | 32.4987 | 0.8940 | 1        | gaussian_gray_denoising_sigma15.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma15](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma15.py) | Urban100 | 34.4336 | 0.9419 | 1        | gaussian_gray_denoising_sigma15.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma15](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma15.py) | Set12    | 33.9642 | 0.9153 | 1        | gaussian_gray_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_gray_sigma15](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma15.py) | BSD68    | 30.4941 | 0.8040 | 1        | gaussian_gray_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_gray_sigma15](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma15.py) | Urban100 | 34.3152 | 0.9409 | 1        | gaussian_gray_denoising_blind.pth   | -        |

**Test Grayscale Gaussian Noise: $\\sigma$=25**

| Method                                                                                                                         | Dataset  | PSNR    | SSIM   | GPU Info | ckpt                                | Download |
| ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------- | ------ | -------- | ----------------------------------- | -------- |
| [restormer_official_gaussian_denoising_gray_sigma25](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma25.py) | Set12    | 31.7289 | 0.8811 | 1        | gaussian_gray_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma25](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma25.py) | BSD68    | 30.1613 | 0.8370 | 1        | gaussian_gray_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma25](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma25.py) | Urban100 | 32.1162 | 0.9140 | 1        | gaussian_gray_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma25](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma25.py) | Set12    | 31.7106 | 0.8810 | 1        | gaussian_gray_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_gray_sigma25](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma25.py) | BSD68    | 28.0652 | 0.7108 | 1        | gaussian_gray_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_gray_sigma25](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma25.py) | Urban100 | 32.0457 | 0.9131 | 1        | gaussian_gray_denoising_blind.pth   | -        |

**Test Grayscale Gaussian Noise: $\\sigma$=50**

| Method                                                                                                                         | Dataset  | PSNR    | SSIM   | GPU Info | ckpt                                | Download |
| ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------- | ------ | -------- | ----------------------------------- | -------- |
| [restormer_official_gaussian_denoising_gray_sigma50](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma50.py) | Set12    | 28.6269 | 0.8188 | 1        | gaussian_gray_denoising_sigma50.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma50](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma50.py) | BSD68    | 27.3266 | 0.7434 | 1        | gaussian_gray_denoising_sigma50.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma50](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma50.py) | Urban100 | 28.9636 | 0.8571 | 1        | gaussian_gray_denoising_sigma50.pth | -        |
| [restormer_official_gaussian_denoising_gray_sigma50](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma50.py) | Set12    | 28.6614 | 0.8197 | 1        | gaussian_gray_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_gray_sigma50](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma50.py) | BSD68    | 25.2580 | 0.5736 | 1        | gaussian_gray_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_gray_sigma50](/configs/restormer/restormer_official_gaussian_denoising_gray_sigma50.py) | Urban100 | 28.9848 | 0.8571 | 1        | gaussian_gray_denoising_blind.pth   | -        |

**Test Color Gaussian Noise: $\\sigma$=15**

| Method                                                                                                                          | Dataset  | PSNR    | SSIM   | GPU Info | ckpt                                 | Download |
| ------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | ------ | -------- | ------------------------------------ | -------- |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | CBSD68   | 34.3506 | 0.9352 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | Kodak24  | 35.4900 | 0.9312 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | McMaster | 35.6072 | 0.9352 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | Urban100 | 35.1522 | 0.9530 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | CBSD68   | 34.3422 | 0.9356 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | Kodak24  | 35.4544 | 0.9308 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | McMaster | 35.5473 | 0.9344 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma15](/configs/restormer/restormer_official_gaussian_denoising_color_sigma15.py) | Urban100 | 35.0754 | 0.9524 | 1        | gaussian_color_denoising_blind.pth   | -        |

**Test Color Gaussian Noise: $\\sigma$=25**

| Method                                                                                                                          | Dataset  | PSNR    | SSIM   | GPU Info | ckpt                                 | Download |
| ------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | ------ | -------- | ------------------------------------ | -------- |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | CBSD68   | 31.7457 | 0.8942 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | Kodak24  | 33.0489 | 0.8943 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | McMaster | 33.3260 | 0.9066 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | Urban100 | 32.9670 | 0.9317 | 1        | gaussian_color_denoising_sigma25.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | CBSD68   | 31.7391 | 0.8945 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | Kodak24  | 33.0380 | 0.8941 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | McMaster | 33.3040 | 0.9063 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma25](/configs/restormer/restormer_official_gaussian_denoising_color_sigma25.py) | Urban100 | 32.9165 | 0.9312 | 1        | gaussian_color_denoising_blind.pth   | -        |

**Test Color Gaussian Noise: $\\sigma$=50**

| Method                                                                                                                          | Dataset  | PSNR    | SSIM   | GPU Info | ckpt                                 | Download |
| ------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | ------ | -------- | ------------------------------------ | -------- |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | CBSD68   | 28.5569 | 0.8127 | 1        | gaussian_color_denoising_sigma50.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | Kodak24  | 30.0122 | 0.8238 | 1        | gaussian_color_denoising_sigma50.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | McMaster | 30.2608 | 0.8515 | 1        | gaussian_color_denoising_sigma50.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | Urban100 | 30.0230 | 0.8902 | 1        | gaussian_color_denoising_sigma50.pth | -        |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | CBSD68   | 28.5582 | 0.8126 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | Kodak24  | 30.0074 | 0.8233 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | McMaster | 30.2671 | 0.8520 | 1        | gaussian_color_denoising_blind.pth   | -        |
| [restormer_official_gaussian_denoising_color_sigma50](/configs/restormer/restormer_official_gaussian_denoising_color_sigma50.py) | Urban100 | 30.0172 | 0.8898 | 1        | gaussian_color_denoising_blind.pth   | -        |

### **Defocus Deblurring**

| Method                                                                                                  | Datasets | PSNR    | SSIM   | MAE    | GPU Info | Download |
| ------------------------------------------------------------------------------------------------------- | -------- | ------- | ------ | ------ | -------- | -------- |
| [restormer_official_defocus_deblur_SingleDPDD](/configs/restormer/restormer_official_defocus_deblur.py) | DPDD_S   | 25.9805 | 0.8166 | 0.0378 | 1        | -        |
| [restormer_official_defocus_deblur_DualDPDD](/configs/restormer/restormer_official_defocus_deblur.py)   | DPDD_D   | 26.6160 | 0.8346 | 0.0354 | 1        | -        |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

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

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
# Motion Deblurring
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_deblur.py Motion_Deblurring_checkpoint_path

# single-gpu test
# Motion Deblurring
python tools/test.py configs/restormer/restormer_official_deblur.py Motion_Deblurring_checkpoint_path

# multi-gpu test
# Motion Deblurring
./tools/dist_test.sh configs/restormer/restormer_official_deblur.py Motion_Deblurring_checkpoint_path

```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

## Citation

```bibtex
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```
