# 教程 5：使用评价指标

MMagic支持**17个指标**以评估模型质量。

有关用法，请参阅[MMagic中的训练与测试](../user_guides/train_test.md)。

在这里，我们将逐个介绍不同指标的详细信息。

本文的结构如下:

01. [MAE](#mae)
02. [MSE](#mse)
03. [PSNR](#psnr)
04. [SNR](#snr)
05. [SSIM](#ssim)
06. [NIQE](#niqe)
07. [SAD](#sad)
08. [MattingMSE](#mattingmse)
09. [GradientError](#gradienterror)
10. [ConnectivityError](#connectivityerror)
11. [FID and TransFID](#fid-and-transfid)
12. [IS and TransIS](#is-and-transis)
13. [Precision and Recall](#precision-and-recall)
14. [PPL](#ppl)
15. [SWD](#swd)
16. [MS-SSIM](#ms-ssim)
17. [Equivarience](#equivarience)

## MAE

MAE是图像的平均绝对误差。
要使用MAE进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='MAE'),
]
```

## MSE

MSE是图像的均方误差。
要使用MSE进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='MSE'),
]
```

## PSNR

PSNR是峰值信噪比。我们的实现方法来自https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio。
要使用PSNR进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='PSNR'),
]
```

## SNR

SNR是信噪比。我们的实现方法来自 https://en.wikipedia.org/wiki/Signal-to-noise_ratio。
要使用SNR进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='SNR'),
]
```

## SSIM

SSIM是图像的结构相似度，在[图像质量评估:从错误可见性到结构相似度](https://live.ece.utexas.edu/publications/2004/zwang_ssim_ieeeip2004.pdf)中提出。我们实现的结果与https://ece.uwaterloo.ca/~z70wang/research/ssim/官方发布的MATLAB代码相同。
要使用SSIM进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='SSIM'),
]
```

## NIQE

NIQE是自然图像质量评估指标，在[制作'完全盲'图像质量分析仪](http://www.live.ece.utexas.edu/publications/2013/mittal2013.pdf)中提出。我们的实现可以产生几乎与官方MATLAB代码相同的结果:http://live.ece.utexas.edu/research/quality/niqe_release.zip。
要使用NIQE进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='NIQE'),
]
```

## SAD

SAD是图像抠图的绝对误差和。该指标计算每个像素的绝对差和所有像素的总和。
要使用SAD进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='SAD'),
]
```

## MattingMSE

MattingMSE是图像抠图的均方误差。
要使用MattingMSE进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='MattingMSE'),
]
```

## GradientError

GradientError是用于评估alpha matte预测的梯度误差。
要使用GradientError进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='GradientError'),
]
```

## ConnectivityError

ConnectivityError是用于评估alpha matte预测的连通性误差。
要使用ConnectivityError进行评估，请在配置文件中添加以下配置:

```python
val_evaluator = [
    dict(type='ConnectivityError'),
]
```

## FID 和 TransFID

Fréchet初始距离是两个图像数据集之间相似度的度量。它被证明与人类对视觉质量的判断有很好的相关性，最常用于评估生成对抗网络样本的质量。FID是通过计算两个高斯函数之间的Fréchet距离来计算的，这些高斯函数适合于Inception网络的特征表示。

在`MMagic`中，我们提供了两个版本的FID计算。一个是常用的PyTorch版本，另一个用于StyleGAN。同时，我们在StyleGAN2-FFHQ1024模型中比较了这两种实现之间的差异(详细信息可以在这里找到\[https://github.com/open-mmlab/mmagic/blob/main/configs/styleganv2/README.md\])。幸运的是，最终结果只是略有不同。因此，我们建议用户采用更方便的PyTorch版本。

**关于PyTorch版本和Tero版本:** 常用的PyTorch版本采用修改后的InceptionV3网络提取真假图像特征。然而，Tero的FID需要Tensorflow InceptionV3的[脚本模块](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt)。注意，应用此脚本模块需要' PyTorch >= 1.6.0 '。

**关于提取真实的初始数据:** 为了方便用户，在测试时自动提取真实的特征并保存在本地，存储的特征在下次测试时自动读取。具体来说，我们将根据用于计算实际特性的参数计算一个哈希值，并使用哈希值来标记特性文件，在测试时，如果' inception_pkl '没有设置，我们将在' MMAGIC_CACHE_DIR ' (~/.cache/openmmlab/mmagic/)中寻找该特性。如果未找到缓存的初始pkl，则将执行提取。

要使用FID指标，请在配置文件中添加以下配置:

```python
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
```

如果您在一台新机器上工作，那么您可以复制'MMAGIC_CACHE_DIR'中的'pkl'文件，将它们复制到新机器并设置'inception_pkl'字段。

```python
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        inception_pkl=
        'work_dirs/inception_pkl/inception_state-capture_mean_cov-full-33ad4546f8c9152e4b3bdb1b0c08dbaf.pkl',  # copied from old machine
        sample_model='ema')
]
```

'TransFID'与'FID'的用法相同，但TransFID是为'Pix2Pix'和'CycleGAN'等翻译模型设计的，适用于我们的评估器。更多信息您可以参考[evaluation](../user_guides/train_test.md)。

## IS 和 TransIS

Inception评分是评估生成图像质量的客观指标，在[改进的训练GANs技术](https://arxiv.org/pdf/1606.03498.pdf)中提出。它使用一个InceptionV3模型来预测生成的图像的类别，并假设: 1)如果图像质量高，它将被归类到特定的类别。2)如果图像具有较高的多样性，则图像的类别范围将很广。因此，条件概率和边际概率的kl -散度可以指示生成图像的质量和多样性。您可以在'metrics.py'中看到完整的实现，它指向https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py。
如果您想使用'IS'指标评估模型，请在配置文件中添加以下配置:

```python
# at the end of the configs/biggan/biggan_2xb25-500kiters_cifar10-32x32.py
metrics = [
    xxx,
    dict(
        type='IS',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
```

需要注意的是，Inception V3的选择和图像大小的调整方法会显著影响最终的IS评分。因此，我们强烈建议用户可以下载[Tero's script model of Inception V3](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt)(加载此脚本模型需要torch >= 1.6)，并使用'Bicubic'插值与'Pillow'后端。

对应于config，您可以设置'resize_method'和'use_pillow_resize'用于图像大小的调整。您也可以将'inception_style'设置为'StyleGAN'用于推荐的tero的初始模型，或'PyTorch'用于torchvision的实现。对于没有互联网的环境，您可以下载初始的权重，并将'inception_path'设置为您的初始模型。

我们还调查了数据加载管线和预训练的Inception V3版本对IS结果的影响。所有IS都在同一组图像上进行评估，这些图像是从ImageNet数据集中随机选择的。

<details> <summary> 显示对比结果 </summary>

|                            Code Base                            | Inception V3 Version | Data Loader Backend | Resize Interpolation Method |          IS           |
| :-------------------------------------------------------------: | :------------------: | :-----------------: | :-------------------------: | :-------------------: |
|   [OpenAI (baseline)](https://github.com/openai/improved-gan)   |      Tensorflow      |       Pillow        |       Pillow Bicubic        | **312.255 +/- 4.970** |
| [StyleGAN-Ada](https://github.com/NVlabs/stylegan2-ada-pytorch) | Tero's Script Model  |       Pillow        |       Pillow Bicubic        |   311.895 +/ 4.844    |
|                          mmagic (Ours)                          |  Pytorch Pretrained  |         cv2         |        cv2 Bilinear         |   322.932 +/- 2.317   |
|                          mmagic (Ours)                          |  Pytorch Pretrained  |         cv2         |         cv2 Bicubic         |   324.604 +/- 5.157   |
|                          mmagic (Ours)                          |  Pytorch Pretrained  |         cv2         |       Pillow Bicubic        |   318.161 +/- 5.330   |
|                          mmagic (Ours)                          |  Pytorch Pretrained  |       Pillow        |       Pillow Bilinear       |   313.126 +/- 5.449   |
|                          mmagic (Ours)                          |  Pytorch Pretrained  |       Pillow        |        cv2 Bilinear         |    318.021+/-3.864    |
|                          mmagic (Ours)                          |  Pytorch Pretrained  |       Pillow        |       Pillow Bicubic        |   317.997 +/- 5.350   |
|                          mmagic (Ours)                          | Tero's Script Model  |         cv2         |        cv2 Bilinear         |   318.879 +/- 2.433   |
|                          mmagic (Ours)                          | Tero's Script Model  |         cv2         |         cv2 Bicubic         |   316.125 +/- 5.718   |
|                          mmagic (Ours)                          | Tero's Script Model  |         cv2         |       Pillow Bicubic        | **312.045 +/- 5.440** |
|                          mmagic (Ours)                          | Tero's Script Model  |       Pillow        |       Pillow Bilinear       |   308.645 +/- 5.374   |
|                          mmagic (Ours)                          | Tero's Script Model  |       Pillow        |       Pillow Bicubic        |   311.733 +/- 5.375   |

</details>

'TransIS'与'IS'的用法相同，但TransIS是为'Pix2Pix'和'CycleGAN'这样的翻译模型设计的，这是为我们的评估器改编的。更多信息可参考[evaluation](../user_guides/train_test.md)。

## Precision and Recall

我们的'Precision and Recall'实现遵循StyleGAN2中使用的版本。在该度量中，采用VGG网络对图像进行特征提取。不幸的是，我们还没有发现PyTorch VGG实现与StyleGAN2中使用的Tero版本产生类似的结果。(关于差异，请参阅这个[文件](https://github.com/open-mmlab/mmagicing/blob/main/configs/styleganv2/README.md)。)因此，在我们的实现中，我们默认采用[Teor's VGG](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt)网络。需要注意的是，应用这个脚本模块需要'PyTorch >= 1.6.0'。如果使用较低的PyTorch版本，我们将使用PyTorch官方VGG网络进行特征提取。
要使用' P&R '进行评估，请在配置文件中添加以下配置:

```python
metrics = [
    dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K')
]
```

## PPL

当在两个随机输入之间进行插值时，感知路径长度测量连续图像（其VGG16嵌入）之间的差异。剧烈的变化意味着多个特征一起发生了变化，它们可能会叠加在一起。通过实验表明，较小的PPL分数表明整体图像质量较高。
作为该指标的基础，我们使用基于感知的成对图像距离，该距离被计算为两个VGG16嵌入之间的加权差，其中权重被拟合，从而评价指标与人类的感知相似性判断一致。
如果我们将潜在空间插值路径细分为线性段，我们可以将该分段路径的总感知长度定义为每个段上感知差异的总和，并且感知路径长度的自然定义将是无限细分下的总和的极限，但在实践中，我们使用一个小的细分`` $`\epsilon=10^｛-4｝`$ ``来近似它。
因此，潜在`space`Z中所有可能端点的平均感知路径长度为

`` $$`L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]`$$ ``

以类似的方式计算潜在 `space` W中的平均感知路径长度：:

`` $$`L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]`$$ ``

当 `` $`z_1, z_2 \sim P(z)`$ ``， 如果我们设置 `sampling` 为 ` full`，  则 `` $` t \sim U(0,1)`$ ``， 如果设置 `sampling` 为 `end`，则`` $` t \in \{0,1\}`$ ``。 `` $` G`$ `` 是生成器(i.e. `` $` g \circ f`$ `` 用于style-based网络)， `` $` d(.,.)`$ `` 用于计算结果图像之间的感知距离。我们通过取100,000个样本来计算期望(在代码中将' num_images '设置为50,000)。

您可以在'metrics.py'中找到完整的实现，参考https://github.com/rosinality/stylegan2-pytorch/blob/master/ppl.py。
如果您想使用'PPL'指标评估模型，请在配置文件中添加以下配置:

```python
# at the end of the configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py
metrics = [
    xxx,
    dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
]
```

## SWD

切片Wasserstein距离是概率分布的差异度量，距离越小表示生成的图像越真实。我们获得每个图像的拉普拉斯金字塔，并从拉普拉斯金字塔中提取小块作为描述符，然后可以通过获取真实和伪描述符切片的Wasserstein距离来计算SWD。
您可以在'metrics.py'中看到完整的实现，参考https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py。
如果您想使用'SWD'指标评估模型，请在配置文件中添加以下配置:

```python
# at the end of the configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
metrics = [
    dict(
        type='SWD',
        prefix='swd',
        fake_nums=16384,
        sample_model='orig',
        image_shape=(3, 64, 64))
]
```

## MS-SSIM

采用多尺度结构相似度来衡量两幅图像的相似度。我们在这里使用MS-SSIM来衡量生成图像的多样性，MS-SSIM得分低表示生成图像的多样性高。您可以在'metrics.py'中看到完整的实现，参考https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py。
如果您想使用'MS-SSIM'指标评估模型，请在配置文件中添加以下配置:

```python
# at the end of the configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
metrics = [
    dict(
        type='MS_SSIM', prefix='ms-ssim', fake_nums=10000,
        sample_model='orig')
]
```

## Equivarience

生成模型的等价性是指模型正变换和几何变换的互换性。目前这个指标只针对StyleGANv3计算，您可以在'metrics.py'中看到完整的实现，参考https://github.com/NVlabs/stylegan3/blob/main/metrics/equivariance.py。
如果您想使用'Equivarience'指标评估模型，请在配置文件中添加以下配置:

```python
# at the end of the configs/styleganv3/stylegan3-t_gamma2.0_8xb4-fp16-noaug_ffhq-256x256.py
metrics = [
    dict(
        type='Equivariance',
        fake_nums=50000,
        sample_mode='ema',
        prefix='EQ',
        eq_cfg=dict(
            compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True))
]
```
