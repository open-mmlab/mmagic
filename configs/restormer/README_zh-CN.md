# Restormer (CVPR'2022)

> [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)

> **任务**: 图像去噪，图像去模糊，图像去雨

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/49083766/206964466-95de972a-ff92-4493-9097-73118590d78f.png" width="800"/>
</div >

## **各个任务下模型的测试结果**

### **图像去雨**

所有数据集均在Y通道上进行测试，测试指标为PSNR和SSIM。

|                方法                | Rain100H<br>PSNR/SSIM (Y) | Rain100L<br>PSNR/SSIM (Y) | Test100<br>PSNR/SSIM (Y) | Test1200<br>PSNR/SSIM (Y) | Test2800<br>PSNR/SSIM (Y) | GPU信息 |                下载                 |
| :--------------------------------: | :-----------------------: | :-----------------------: | :----------------------: | :-----------------------: | :-----------------------: | :-----: | :---------------------------------: |
| [restormer_official_rain13k](./restormer_official_rain13k.py) |      31.4804/0.9056       |      39.1023/0.9787       |      32.0287/0.9239      |      33.2251/0.9272       |      34.2170/0.9451       |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth) \| log |

### **图像去模糊**

Gopro和HIDE数据集上使用RGB通道测试，ReakBlur-J 和 ReakBlur-R数据集使用Y通道测试。测试指标为PSNR和SSIM。

|                      方法                      | GoPro<br>PSNR/SSIM (RGB) | HIDE<br>PSNR/SSIM (RGB) | RealBlur-J<br>PSNR/SSIM (Y) | RealBlur-R<br>PSNR/SSIM (Y) | GPU信息 |                      下载                      |
| :--------------------------------------------: | :----------------------: | :---------------------: | :-------------------------: | :-------------------------: | :-----: | :--------------------------------------------: |
| [restormer_official_gopro](./restormer_official_gopro.py) |      32.9295/0.9496      |     31.2289/0.9345      |       28.4356/0.8681        |       35.9141/0.9707        |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth) \| log |

### **图像去失焦模糊**

所有指标均在RGB通道上进行测试。测试指标为PSNR、SSIM、MAE和LPIPS.

|  方法  | 室内场景图像的PSNR | 室内场景图像的SSIM | 室内场景图像的MAE | 室内场景图像的LPIPS | 室外场景图像的PSNR | 室外场景图像的SSIM | 室外场景图像的MAE | 室外场景图像的LPIPS | 所有图像平均PSNR | 所有图像平均SSIM | 所有图像平均MAE | 所有图像平均LPIPS | GPU 信息 |  下载   |
| :----: | :-------------: | :-------------: | :------------: | :--------------: | :-------------: | :-------------: | :------------: | :--------------: | :------------: | :-------------: | :------------: | :--------------: | :------: | :-----: |
| [restormer_official_dpdd-single](./restormer_official_dpdd-single.py) |     28.8681     |     0.8859      |     0.0251     |        -         |     23.2410     |     0.7509      |     0.0499     |        -         |    25.9805     |     0.8166      |     0.0378     |        -         |    1     | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth) \| log |
| [restormer_official_dpdd-dual](./restormer_official_dpdd-dual.py) |     26.6160     |     0.8346      |     0.0354     |        -         |     26.6160     |     0.8346      |     0.0354     |        -         |    26.6160     |     0.8346      |     0.0354     |        -         |    1     | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth) \| log |

### **图像高斯噪声去除**

**灰度图的高斯噪声**

使用PSNR和SSIM指标对数据集上的灰度图进行测试。

|                              方法                               | $\\sigma$ | Set12<br>PSNR/SSIM | BSD68<br>PSNR/SSIM | Urban100<br>PSNR/SSIM | GPU信息 |                               下载                               |
| :-------------------------------------------------------------: | :-------: | :----------------: | :----------------: | :-------------------: | :-----: | :--------------------------------------------------------------: |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) |    15     |   34.0182/0.9160   |   32.4987/0.8940   |    34.4336/0.9419     |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth) \| log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) |    25     |   31.7289/0.8811   |   30.1613/0.8370   |    32.1162/0.9140     |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth) \| log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) |    50     |   28.6269/0.8188   |   27.3266/0.7434   |    28.9636/0.8571     |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth) \| log |
|                                                                 |           |                    |                    |                       |         |                                                                  |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) |    15     |   33.9642/0.9153   |   32.4994/0.8928   |    34.3152/0.9409     |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) |    25     |   31.7106/0.8810   |   30.1486/0.8360   |    32.0457/0.9131     |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) |    50     |   28.6614/0.8197   |   27.3537/0.7422   |    28.9848/0.8571     |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |

> 上面三行代表每个噪声等级训练一个单独的模型，下面三行代表学习一个单一的模型来处理各种噪音水平。

**彩色图像的高斯噪声**

所有指标均在RGB通道上进行测试，测试指标为PSNR和SSIM。

|                  方法                   | $\\sigma$ | CBSD68<br>PSNR/SSIM (RGB) | Kodak24<br>PSNR/SSIM (RGB) | McMaster<br>PSNR/SSIM (RGB) | Urban100<br>PSNR/SSIM (RGB) | GPU信息 |                   下载                   |
| :-------------------------------------: | :-------: | :-----------------------: | :------------------------: | :-------------------------: | :-------------------------: | :-----: | :--------------------------------------: |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py) |    15     |      34.3506/0.9352       |       35.4900/0.9312       |       35.6072/0.9352        |       35.1522/0.9530        |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth) \| log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py) |    25     |      31.7457/0.8942       |       33.0489/0.8943       |       33.3260/0.9066        |       32.9670/0.9317        |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth) \| log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py) |    50     |      28.5569/0.8127       |       30.0122/0.8238       |       30.2608/0.8515        |       30.0230/0.8902        |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth) \| log |
|                                         |           |                           |                            |                             |                             |         |                                          |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py) |    15     |      34.3422/0.9356       |       35.4544/0.9308       |       35.5473/0.9344        |       35.0754/0.9524        |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) \| log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py) |    25     |      31.7391/0.8945       |       33.0380/0.8941       |       33.3040/0.9063        |       32.9165/0.9312        |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) \| log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py) |    50     |      28.5582/0.8126       |       30.0074/0.8233       |       30.2671/0.8520        |       30.0172/0.8898        |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) \| log |

> 上面三行代表每个噪声等级训练一个单独的模型，下面三行代表学习一个单一的模型来处理各种噪音水平。

### **真实场景图像去噪**

所有指标均在RGB通道上进行测试，测试指标为PSNR和SSIM。

|                          方法                           | SIDD<br>PSNR/SSIM | GPU信息 |                                                  下载                                                   |
| :-----------------------------------------------------: | :---------------: | :-----: | :-----------------------------------------------------------------------------------------------------: |
| [restormer_official_sidd](./restormer_official_sidd.py) |  40.0156/0.9225   |    1    | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_sidd-9e7025db.pth) \| log |

## 使用方法

**训练**

可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# cpu test
# Deraining
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_rain13k.py https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth

# Motion Deblurring
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_gopro.py https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth

# Defocus Deblurring
# Single
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dpdd-dual.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth
# Dual
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dpdd-single.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth

# Gaussian Denoising
# Test Grayscale Gaussian Noise
# sigma15
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma25
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma50
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# Test Color Gaussian Noise
# sigma15
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma25
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma50
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# single-gpu test
# Deraining
python tools/test.py configs/restormer/restormer_official_rain13k.py https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth

# Motion Deblurring
python tools/test.py configs/restormer/restormer_official_gopro.py https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth

# Defocus Deblurring
# Single
python tools/test.py configs/restormer/restormer_official_dpdd-dual.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth
# Dual
python tools/test.py configs/restormer/restormer_official_dpdd-single.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth

# Gaussian Denoising
# Test Grayscale Gaussian Noise
# sigma15
python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth

python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma25
python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth

python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma50
python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth

python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# Test Color Gaussian Noise
# sigma15
python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth

python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma25
python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth

python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma50
python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth

python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth


# multi-gpu test
# Deraining
./tools/dist_test.sh configs/restormer/restormer_official_rain13k.py https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth

# Motion Deblurring
./tools/dist_test.sh configs/restormer/restormer_official_gopro.py https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth

# Defocus Deblurring
# Single
./tools/dist_test.sh configs/restormer/restormer_official_dpdd-dual.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth
# Dual
./tools/dist_test.sh configs/restormer/restormer_official_dpdd-single.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth

# Gaussian Denoising
# Test Grayscale Gaussian Noise
# sigma15
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma25
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma50
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# Test Color Gaussian Noise
# sigma15
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma25
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma50
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
