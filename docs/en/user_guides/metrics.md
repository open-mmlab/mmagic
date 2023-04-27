# Tutorial 5: Using metrics in MMagic

MMagic supports **17 metrics** to assess the quality of models.

Please refer to [Train and Test in MMagic](../user_guides/train_test.md) for usages.

Here, we will specify the details of different metrics one by one.

The structure of this guide are as follows:

- [Tutorial 5: Using metrics in MMagic](#tutorial-5-using-metrics-in-mmagic)
  - [MAE](#mae)
  - [MSE](#mse)
  - [PSNR](#psnr)
  - [SNR](#snr)
  - [SSIM](#ssim)
  - [NIQE](#niqe)
  - [SAD](#sad)
  - [MattingMSE](#mattingmse)
  - [GradientError](#gradienterror)
  - [ConnectivityError](#connectivityerror)
  - [FID and TransFID](#fid-and-transfid)
  - [IS and TransIS](#is-and-transis)
  - [Precision and Recall](#precision-and-recall)
  - [PPL](#ppl)
  - [SWD](#swd)
  - [MS-SSIM](#ms-ssim)
  - [Equivarience](#equivarience)

## MAE

MAE is Mean Absolute Error metric for image.
To evaluate with MAE, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='MAE'),
]
```

## MSE

MSE is Mean Squared Error metric for image.
To evaluate with MSE, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='MSE'),
]
```

## PSNR

PSNR is Peak Signal-to-Noise Ratio. Our implement refers to https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio.
To evaluate with PSNR, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='PSNR'),
]
```

## SNR

SNR is Signal-to-Noise Ratio. Our implementation refers to https://en.wikipedia.org/wiki/Signal-to-noise_ratio.
To evaluate with SNR, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='SNR'),
]
```

## SSIM

SSIM is structural similarity for image, proposed in [Image quality assessment: from error visibility to structural similarity](https://live.ece.utexas.edu/publications/2004/zwang_ssim_ieeeip2004.pdf). The results of our implementation are the same as that of the official released MATLAB code in https://ece.uwaterloo.ca/~z70wang/research/ssim/.
To evaluate with SSIM, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='SSIM'),
]
```

## NIQE

NIQE is Natural Image Quality Evaluator metric, proposed in [Making a "Completely Blind" Image Quality Analyzer](http://www.live.ece.utexas.edu/publications/2013/mittal2013.pdf). Our implementation could produce almost the same results as the official MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip.

To evaluate with NIQE, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='NIQE'),
]
```

## SAD

SAD is Sum of Absolute Differences metric for image matting. This metric compute per-pixel absolute difference and sum across all pixels.
To evaluate with SAD, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='SAD'),
]
```

## MattingMSE

MattingMSE is Mean Squared Error metric for image matting.
To evaluate with MattingMSE, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='MattingMSE'),
]
```

## GradientError

GradientError is Gradient error for evaluating alpha matte prediction.
To evaluate with GradientError, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='GradientError'),
]
```

## ConnectivityError

ConnectivityError is Connectivity error for evaluating alpha matte prediction.
To evaluate with ConnectivityError, please add the following configuration in the config file:

```python
val_evaluator = [
    dict(type='ConnectivityError'),
]
```

## FID and TransFID

Fréchet Inception Distance is a measure of similarity between two datasets of images. It was shown to correlate well with the human judgment of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

In `MMagic`, we provide two versions for FID calculation. One is the commonly used PyTorch version and the other one is used in StyleGAN paper. Meanwhile, we have compared the difference between these two implementations in the StyleGAN2-FFHQ1024 model (the details can be found [here](https://github.com/open-mmlab/mmagic/blob/main/configs/styleganv2/README.md)). Fortunately, there is a marginal difference in the final results. Thus, we recommend users adopt the more convenient PyTorch version.

**About PyTorch version and Tero's version:** The commonly used PyTorch version adopts the modified InceptionV3 network to extract features for real and fake images. However, Tero's FID requires a [script module](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for Tensorflow InceptionV3. Note that applying this script module needs `PyTorch >= 1.6.0`.

**About extracting real inception data:** For the users' convenience, the real features will be automatically extracted at test time and saved locally, and the stored features will be automatically read at the next test. Specifically, we will calculate a hash value based on the parameters used to calculate the real features, and use the hash value to mark the feature file, and when testing, if the `inception_pkl` is not set, we will look for the feature in `MMAGIC_CACHE_DIR` (~/.cache/openmmlab/mmagic/). If cached inception pkl is not found, then extracting will be performed.

To use the FID metric, you should add the metric in a config file like this:

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

If you work on an new machine, then you can copy the `pkl` files in `MMAGIC_CACHE_DIR` and copy them to new machine and set `inception_pkl` field.

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

`TransFID` has same usage as `FID`, but it's designed for translation models like `Pix2Pix` and `CycleGAN`, which is adapted for our evaluator. You can refer
to [evaluation](../user_guides/train_test.md) for details.

## IS and TransIS

Inception score is an objective metric for evaluating the quality of generated images, proposed in [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf). It uses an InceptionV3 model to predict the class of the generated images, and suppose that 1) If an image is of high quality, it will be categorized into a specific class. 2) If images are of high diversity, the range of images' classes will be wide. So the KL-divergence of the conditional probability and marginal probability can indicate the quality and diversity of generated images. You can see the complete implementation in `metrics.py`, which refers to https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py.
If you want to evaluate models with `IS` metrics, you can add the `metrics` into your config file like this:

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

To be noted that, the selection of Inception V3 and image resize method can significantly influence the final IS score. Therefore, we strongly recommend users may download the [Tero's script model of Inception V3](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) (load this script model need torch >= 1.6) and use `Bicubic` interpolation with `Pillow` backend.

Corresponding to config, you can set `resize_method` and `use_pillow_resize` for image resizing. You can also set `inception_style` as `StyleGAN` for recommended tero's inception model, or `PyTorch` for torchvision's implementation. For environment without internet, you can download the inception's weights, and set `inception_path` to your inception model.

We also perform a survey on the influence of data loading pipeline and the version of pretrained Inception V3 on the IS result. All IS are evaluated on the same group of images which are randomly selected from the ImageNet dataset.

<details> <summary> Show the Comparison Results </summary>

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

`TransIS` has same usage as `IS`, but it's designed for translation models like `Pix2Pix` and `CycleGAN`, which is adapted for our evaluator. You can refer
to [evaluation](../user_guides/train_test.md) for details.

## Precision and Recall

Our `Precision and Recall` implementation follows the version used in StyleGAN2. In this metric, a VGG network will be adopted to extract the features for images. Unfortunately, we have not found a PyTorch VGG implementation leading to similar results with Tero's version used in StyleGAN2. (About the differences, please see this [file](https://github.com/open-mmlab/mmagic/blob/main/configs/styleganv2/README.md).) Thus, in our implementation, we adopt [Teor's VGG](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) network by default. Importantly, applying this script module needs `PyTorch >= 1.6.0`. If with a lower PyTorch version, we will use the PyTorch official VGG network for feature extraction.

To evaluate with `P&R`, please add the following configuration in the config file:

```python
metrics = [
    dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K')
]
```

## PPL

Perceptual path length measures the difference between consecutive images (their VGG16 embeddings) when interpolating between two random inputs. Drastic changes mean that multiple features have changed together and that they might be entangled. Thus, a smaller PPL score appears to indicate higher overall image quality by experiments. \
As a basis for our metric, we use a perceptually-based pairwise image distance that is calculated as a weighted difference between two VGG16 embeddings, where the weights are fit so that the metric agrees with human perceptual similarity judgments.
If we subdivide a latent space interpolation path into linear segments, we can define the total perceptual length of this segmented path as the sum of perceptual differences over each segment, and a natural definition for the perceptual path length would be the limit of this sum under infinitely fine subdivision, but in practice we approximate it using a small subdivision `` $`\epsilon=10^{-4}`$ ``.
The average perceptual path length in latent `space` Z, over all possible endpoints, is therefore

`` $$`L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]`$$ ``

Computing the average perceptual path length in latent `space` W is carried out in a similar fashion:

`` $$`L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]`$$ ``

Where `` $`z_1, z_2 \sim P(z)`$ ``, and `` $` t \sim U(0,1)`$ `` if we set `sampling` to full, `` $` t \in \{0,1\}`$ `` if we set `sampling` to end. `` $` G`$ `` is the generator(i.e. `` $` g \circ f`$ `` for style-based networks), and `` $` d(.,.)`$ `` evaluates the perceptual distance between the resulting images.We compute the expectation by taking 100,000 samples (set `num_images` to 50,000 in our code).

You can find the complete implementation in `metrics.py`, which refers to https://github.com/rosinality/stylegan2-pytorch/blob/master/ppl.py.
If you want to evaluate models with `PPL` metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py
metrics = [
    xxx,
    dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
]
```

## SWD

Sliced Wasserstein distance is a discrepancy measure for probability distributions, and smaller distance indicates generated images look like the real ones. We obtain the Laplacian pyramids of every image and extract patches from the Laplacian pyramids as descriptors, then SWD can be calculated by taking the sliced Wasserstein distance of the real and fake descriptors.
You can see the complete implementation in `metrics.py`, which refers to https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py.
If you want to evaluate models with `SWD` metrics, you can add the `metrics` into your config file like this:

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

Multi-scale structural similarity is used to measure the similarity of two images. We use MS-SSIM here to measure the diversity of generated images, and a low MS-SSIM score indicates the high diversity of generated images. You can see the complete implementation in `metrics.py`, which refers to https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py.
If you want to evaluate models with `MS-SSIM` metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
metrics = [
    dict(
        type='MS_SSIM', prefix='ms-ssim', fake_nums=10000,
        sample_model='orig')
]
```

## Equivarience

Equivarience of generative models refer to the exchangeability of model forward and geometric transformations. Currently this metric is only calculated for StyleGANv3,
you can see the complete implementation in `metrics.py`, which refers to https://github.com/NVlabs/stylegan3/blob/main/metrics/equivariance.py.
If you want to evaluate models with `Equivarience` metrics, you can add the `metrics` into your config file like this:

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
