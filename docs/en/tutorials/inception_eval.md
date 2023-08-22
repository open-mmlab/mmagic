# Tutorial 5: How to compute FID and KID for measruing the difference between the distribution of real data and restored-data

<!-- TOC -->

- [Tutorial 5: How to compute FID and KID for measruing the difference between the distribution of real data and restored-data](#tutorial-5-how-to-compute-fid-and-kid-for-measruing-the-difference-between-the-distribution-of-real-data-and-restored-data)
  - [Why FID and KID for image restoration tasks?](#why-fid-and-kid-for-image-restoration-tasks)
- [Set Config File](#set-config-file)

<!-- TOC -->

## Why FID and KID for image restoration tasks?

Commonly used metrics for image/video resoration are PSNR, SSIM or LPIPS which directly compare high-quality original data and restored data.
While these are good metrics to order the restoration performance of different models, they have the requirement that undegraded images should be given.

Recently, some unpaired restoration methods have been proposed to restore real-world images in the condition where only degraded images are accessible.
In this condition, it is difficult to compare the various proposed models with common metrics such as PSNR, SSIM since there is no corresponding ground-truth data.

An alternative way to evaluate these models in real-world settings is to compare the distributions of real data and restored-data, rather than directly comparing corresponding images.

To this end, MMEditing provides functions to compute  *Fréchet inception distance* (FID) and *Kernel Inception Distance* (KID), which are commonly used in image generation tasks to check fidelity of generated images, metrics that measure the difference between the two distributions.
Currently, computing FID and KID is only available for restoration tasks.

## Set Config File

FID and KID can be meausred after images from two distributions are extractes as feature vectors with the InceptionV3 model.

To compute the distance between extracted feature vectors, we can add `FID` and `KID` metric in `test_cfg` as follow:

```python3
test_cfg = dict(
    metrics=[
        'PSNR', 'SSIM', 'FID',
        dict(type='KID', num_repeats=100, sample_size=1000)
    ],
    inception_style='StyleGAN',  # or pytorch
    crop_border=0)
```
