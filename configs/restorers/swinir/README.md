# SwinIR (ICCVW'2021)

> [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Image restoration is a long-standing low-level vision problem that aims to restore high-quality images from low-quality images (e.g., downscaled, noisy and compressed images). While state-of-the-art image restoration methods are based on convolutional neural networks, few attempts have been made with Transformers which show impressive performance on high-level vision tasks. In this paper, we propose a strong baseline model SwinIR for image restoration based on the Swin Transformer. SwinIR consists of three parts: shallow feature extraction, deep feature extraction and high-quality image reconstruction. In particular, the deep feature extraction module is composed of several residual Swin Transformer blocks (RSTB), each of which has several Swin Transformer layers together with a residual connection. We conduct experiments on three representative tasks: image super-resolution (including classical, lightweight and real-world image super-resolution), image denoising (including grayscale and color image denoising) and JPEG compression artifact reduction. Experimental results demonstrate that SwinIR outperforms state-of-the-art methods on different tasks by up to 0.14∼0.45dB, while the total number of parameters can be reduced by up to 67%.

<!-- [IMAGE] -->
<div align=center >
 <img src="https://github.com/JingyunLiang/SwinIR/raw/main/figs/SwinIR_archi.png" width="800"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                            Method                            |       Set5        |      Set14       |      DIV2K       |                           Download                           |
| :----------------------------------------------------------: | :---------------: | :--------------: | :--------------: | :----------------------------------------------------------: |
| [swinir_classical_patch48_2x](/configs/restorers/swinir/swinir_psnr_patch48.py) | - / - | - / - | - / - | [model](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420_112550.log.json) |
| [-](/configs/restorers/swinir/-.py) | - /  - | - / - | - / - | [model](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508_191042.log.json) |


## Citation

```bibtex
@inproceedings{liang2021swinir,
    title={SwinIR: Image Restoration Using Swin Transformer},
    author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
    booktitle={IEEE International Conference on Computer Vision Workshops},
    year={2021}
}
```
