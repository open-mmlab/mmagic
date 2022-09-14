# Design Your Own Metrics

MMEditing supports various metrics to assess the quality of generative models. Refer to [train_test](../user_guides/4_train_test.md) for usages.
Here, we will specify the details of different metrics one by one.

The structure of this guide are as follows:

- [FID and TransFID](#fid-and-transfid)
- [IS and TransIS](#is-and-transis)
- [Precision and Recall](#precision-and-recall)
- [PPL](#ppl)
- [SWD](#swd)
- [MS-SSIM](#ms-ssim)
- [Equivarience](#equivarience)

## FID and TransFID

Fréchet Inception Distance is a measure of similarity between two datasets of images. It was shown to correlate well with the human judgment of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

In `MMEditing`, we provide two versions for FID calculation. One is the commonly used PyTorch version and the other one is used in StyleGAN paper. Meanwhile, we have compared the difference between these two implementations in the StyleGAN2-FFHQ1024 model (the details can be found [here](https://github.com/open-mmlab/mmgeneration/blob/master/configs/styleganv2/README.md)). Fortunately, there is a marginal difference in the final results. Thus, we recommend users adopt the more convenient PyTorch version.

**About PyTorch version and Tero's version:** The commonly used PyTorch version adopts the modified InceptionV3 network to extract features for real and fake images. However, Tero's FID requires a [script module](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for Tensorflow InceptionV3. Note that applying this script module needs `PyTorch >= 1.6.0`.

**About extracting real inception data:** For the users' convenience, the real features will be automatically extracted at test time and saved locally, and the stored features will be automatically read at the next test. Specifically, we will calculate a hash value based on the parameters used to calculate the real features, and use the hash value to mark the feature file, and when testing, if the `inception_pkl` is not set, we will look for the feature in `MMEDIT_CACHE_DIR` (~/.cache/openmmlab/mmedit/). If cached inception pkl is not found, then extracting will be performed.

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

If you work on an new machine, then you can copy the `pkl` files in `MMEDIT_CACHE_DIR` and copy them to new machine and set `inception_pkl` field.

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
to [evaluation](../advanced_guides/evaluation.md) for details.

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

To be noted that, the selection of Inception V3 and image resize method can significantly influence the final IS score. Therefore, we strongly recommend users may download the [Tero's script model of Inception V3](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) (load this script model need torch >= 1.6) and use `Bicubic` interpolation with `Pillow` backend. We provide a template for the [data process pipline](https://github.com/open-mmlab/mmgeneration/tree/master/configs/_base_/datasets/Inception_Score.py) as well.

Corresponding to config, you can set `resize_method` and `use_pillow_resize` for image resizing. You can also set `inception_style` as `StyleGAN` for recommended tero's inception model, or `PyTorch` for torchvision's implementation. For environment without internet, you can download the inception's weights, and set `inception_path` to your inception model.

We also perform a survey on the influence of data loading pipeline and the version of pretrained Inception V3 on the IS result. All IS are evaluated on the same group of images which are randomly selected from the ImageNet dataset.

<details> <summary> Show the Comparison Results </summary>

|                            Code Base                            | Inception V3 Version | Data Loader Backend | Resize Interpolation Method |          IS           |
| :-------------------------------------------------------------: | :------------------: | :-----------------: | :-------------------------: | :-------------------: |
|   [OpenAI (baseline)](https://github.com/openai/improved-gan)   |      Tensorflow      |       Pillow        |       Pillow Bicubic        | **312.255 +/- 4.970** |
| [StyleGAN-Ada](https://github.com/NVlabs/stylegan2-ada-pytorch) | Tero's Script Model  |       Pillow        |       Pillow Bicubic        |   311.895 +/ 4.844    |
|                          mmedit (Ours)                          |  Pytorch Pretrained  |         cv2         |        cv2 Bilinear         |   322.932 +/- 2.317   |
|                          mmedit (Ours)                          |  Pytorch Pretrained  |         cv2         |         cv2 Bicubic         |   324.604 +/- 5.157   |
|                          mmedit (Ours)                          |  Pytorch Pretrained  |         cv2         |       Pillow Bicubic        |   318.161 +/- 5.330   |
|                          mmedit (Ours)                          |  Pytorch Pretrained  |       Pillow        |       Pillow Bilinear       |   313.126 +/- 5.449   |
|                          mmedit (Ours)                          |  Pytorch Pretrained  |       Pillow        |        cv2 Bilinear         |    318.021+/-3.864    |
|                          mmedit (Ours)                          |  Pytorch Pretrained  |       Pillow        |       Pillow Bicubic        |   317.997 +/- 5.350   |
|                          mmedit (Ours)                          | Tero's Script Model  |         cv2         |        cv2 Bilinear         |   318.879 +/- 2.433   |
|                          mmedit (Ours)                          | Tero's Script Model  |         cv2         |         cv2 Bicubic         |   316.125 +/- 5.718   |
|                          mmedit (Ours)                          | Tero's Script Model  |         cv2         |       Pillow Bicubic        | **312.045 +/- 5.440** |
|                          mmedit (Ours)                          | Tero's Script Model  |       Pillow        |       Pillow Bilinear       |   308.645 +/- 5.374   |
|                          mmedit (Ours)                          | Tero's Script Model  |       Pillow        |       Pillow Bicubic        |   311.733 +/- 5.375   |

</details>

`TransIS` has same usage as `IS`, but it's designed for translation models like `Pix2Pix` and `CycleGAN`, which is adapted for our evaluator. You can refer
to [evaluation](../advanced_guides/evaluation.md) for details.

## Precision and Recall

Our `Precision and Recall` implementation follows the version used in StyleGAN2. In this metric, a VGG network will be adopted to extract the features for images. Unfortunately, we have not found a PyTorch VGG implementation leading to similar results with Tero's version used in StyleGAN2. (About the differences, please see this [file](https://github.com/open-mmlab/mmediting/blob/1.x/configs/styleganv2/README.md).) Thus, in our implementation, we adopt [Teor's VGG](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) network by default. Importantly, applying this script module needs `PyTorch >= 1.6.0`. If with a lower PyTorch version, we will use the PyTorch official VGG network for feature extraction.

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
