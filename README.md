<div align="center">
  <img src="docs/en/_static/image/mmediting-logo.png" width="500px"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://badge.fury.io/py/mmedit.svg)](https://pypi.org/project/mmedit/)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmediting.readthedocs.io/en/1.x/)
[![badge](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![license](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/1.x/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)

[📘Documentation](https://mmediting.readthedocs.io/en/1.x/) |
[🛠️Installation](https://mmediting.readthedocs.io/en/1.x/2_get_started.html#installation) |
[👀Model Zoo](https://mmediting.readthedocs.io/en/1.x/3_model_zoo.html) |
[🆕Update News](docs/en/changelog.md) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmediting/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmediting/issues)

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMEditing is an open-source image and video editing&generating toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Currently, MMEditing support the following tasks:

<div align="center">
  <img src="https://user-images.githubusercontent.com/22982797/191167628-2ac529d6-6614-4b53-ad65-0cfff909aa7d.jpg"/>
</div>

The master branch works with **PyTorch 1.5+**.

Some Demos:

https://user-images.githubusercontent.com/12756472/158972852-be5849aa-846b-41a8-8687-da5dee968ac7.mp4

https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">
  <b> GAN Interpolation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114679300-9fd4f900-9d3e-11eb-8f37-c36a018c02f7.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN Projector</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114524392-c11ee200-9c77-11eb-8b6d-37bc637f5626.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN Manipulation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114523716-20302700-9c77-11eb-804e-327ae1ca0c5b.gif" width="200"/>
</div></td>
  </tr>
</thead>
</table>

### Major features

- **Modular design**

  We decompose the editing framework into different components and one can easily construct a customized editor framework by combining different modules.

- **Support of multiple tasks**

  The toolbox directly supports popular and contemporary *inpainting*, *matting*, *super-resolution*, *interpolation* and *generation* tasks.

- **Efficient Distributed Training for Generative Models:**

  With support of [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), distributed training for dynamic architectures can be easily implemented.

- **State of the art**

  The toolbox provides state-of-the-art methods in inpainting/matting/super-resolution/interpolation/generation.

Note that **MMSR** has been merged into this repo, as a part of MMEditing.
With elaborate designs of the new framework and careful implementations,
hope MMEditing could provide better experience.

## What's New

### 🌟 Preview of 1.x version

A brand new version of [**MMEditing v1.0.0rc5**](https://github.com/open-mmlab/mmediting/releases/tag/v1.0.0rc5) was released in 04/01/2023:

- Support well-known text-to-image method [Stable Diffusion](configs/stable_diffusion/README.md)!
- Support an efficient image restoration algorithm [Restormer](configs/restormer/README.md)!
- Support a new text-to-image algorithm [GLIDE](projects/glide/configs/README.md)!
- Support swin based image restoration algorithm [SwinIR](configs/swinir/README.md)!
- [Projects](projects/README.md) is opened for community to add projects to MMEditing.
- Support all the tasks, models, metrics, and losses in [MMGeneration](https://github.com/open-mmlab/mmgeneration) 😍.
- Unifies interfaces of all components based on [MMEngine](https://github.com/open-mmlab/mmengine).
- Support patch-based and slider-based image and video comparison viewer.

Find more new features in [1.x branch](https://github.com/open-mmlab/mmediting/tree/1.x). Issues and PRs are welcome!

### 💎 Stable version

**0.16.0** was released in 31/10/2022:

- `VisualizationHook` is deprecated. Users should use `MMEditVisualizationHook` instead.
- Fix FLAVR register.
- Fix the number of channels in RDB.

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation

MMEditing depends on [PyTorch](https://pytorch.org/), [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

**Step 2.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
# wait for more pre-compiled pkgs to release
mim install 'mmcv>=2.0.0rc1'
```

**Step 3.**
Install MMEditing from source.

```shell
git clone -b 1.x https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

Please refer to [installation](docs/en/get_started/install.md) for more detailed instruction.

## Getting Started

Please see [quick run](docs/en/get_started/quick_run.md) and [inference](docs/en/user_guides/inference.md) for the basic usage of MMEditing.

## Model Zoo

Supported algorithms:

<details open>
<summary>Inpainting</summary>

- ✅ [Global&Local](configs/global_local/README.md) (ToG'2017)
- ✅ [DeepFillv1](configs/deepfillv1/README.md) (CVPR'2018)
- ✅ [PConv](configs/partial_conv/README.md) (ECCV'2018)
- ✅ [DeepFillv2](configs/deepfillv2/README.md) (CVPR'2019)
- ✅ [AOT-GAN](configs/aot_gan/README.md) (TVCG'2021)

</details>

<details open>
<summary>Matting</summary>

- ✅ [DIM](configs/dim/README.md) (CVPR'2017)
- ✅ [IndexNet](configs/indexnet/README.md) (ICCV'2019)
- ✅ [GCA](configs/gca/README.md) (AAAI'2020)

</details>

<details open>
<summary>Image-Super-Resolution</summary>

- ✅ [SRCNN](configs/srcnn/README.md) (TPAMI'2015)
- ✅ [SRResNet&SRGAN](configs/srgan_resnet/README.md) (CVPR'2016)
- ✅ [EDSR](configs/edsr/README.md) (CVPR'2017)
- ✅ [ESRGAN](configs/esrgan/README.md) (ECCV'2018)
- ✅ [RDN](configs/rdn/README.md) (CVPR'2018)
- ✅ [DIC](configs/dic/README.md) (CVPR'2020)
- ✅ [TTSR](configs/ttsr/README.md) (CVPR'2020)
- ✅ [GLEAN](configs/glean/README.md) (CVPR'2021)
- ✅ [LIIF](configs/liif/README.md) (CVPR'2021)
- ✅ [Real-ESRGAN](configs/real_esrgan/README.md) (ICCVW'2021)

</details>

<details open>
<summary>Video-Super-Resolution</summary>

- ✅ [EDVR](configs/edvr/README.md) (CVPR'2019)
- ✅ [TOF](configs/tof/README.md) (IJCV'2019)
- ✅ [TDAN](configs/tdan/README.md) (CVPR'2020)
- ✅ [BasicVSR](configs/basicvsr/README.md) (CVPR'2021)
- ✅ [IconVSR](configs/iconvsr/README.md) (CVPR'2021)
- ✅ [BasicVSR++](configs/basicvsr_pp/README.md) (CVPR'2022)
- ✅ [RealBasicVSR](configs/real_basicvsr/README.md) (CVPR'2022)

</details>

<details open>
<summary>Video Interpolation</summary>

- ✅ [TOFlow](configs/tof/README.md) (IJCV'2019)
- ✅ [CAIN](configs/cain/README.md) (AAAI'2020)
- ✅ [FLAVR](configs/flavr/README.md) (CVPR'2021)

</details>

<details open>
<summary>Image Colorization</summary>

- ✅ [InstColorization](configs/inst_colorization/README.md) (CVPR'2020)

</details>

<details open>
<summary>Unconditional GANs</summary>

- ✅ [DCGAN](configs/dcgan/README.md) (ICLR'2016)
- ✅ [WGAN-GP](configs/wgan-gp/README.md) (NeurIPS'2017)
- ✅ [LSGAN](configs/lsgan/README.md) (ICCV'2017)
- ✅ [GGAN](configs/ggan/README.md) (ArXiv'2017)
- ✅ [PGGAN](configs/pggan/README.md) (ICLR'2018)
- ✅ [StyleGANV1](configs/styleganv1/README.md) (CVPR'2019)
- ✅ [StyleGANV2](configs/styleganv2/README.md) (CVPR'2020)
- ✅ [StyleGANV3](configs/styleganv3/README.md) (NeurIPS'2021)

</details>

<details open>
<summary>Conditional GANs</summary>

- ✅ [SNGAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [Projection GAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [SAGAN](configs/sagan/README.md) (ICML'2019)
- ✅ [BIGGAN/BIGGAN-DEEP](configs/biggan/README.md) (ICLR'2019)

</details>

<details open>
<summary>Image2Image</summary>

- ✅ [Pix2Pix](configs/pix2pix/README.md) (CVPR'2017)
- ✅ [CycleGAN](configs/cyclegan/README.md) (ICCV'2017)

</details>

<details open>
<summary>Internal Learning</summary>

- ✅ [SinGAN](configs/singan/README.md) (ICCV'2019)

</details>

<details open>
<summary>Text2Image</summary>

- ✅ [GLIDE](projects/glide/configs/README.md) (NeurIPS'2021)
- ✅ [Disco-Diffusion](configs/disco_diffusion/README.md)
- ✅ [Stable-Diffusion](configs/stable_diffusion/README.md)

</details>

<details open>

<summary>3D-aware Generation</summary>

- ✅ [EG3D](configs/eg3d/README.md) (CVPR'2022)

</details>

<details open>

<summary>Image Restoration</summary>

- ✅ [SwinIR](configs/swinir/README.md) (ICCVW'2021)
- ✅ [NAFNet](configs/nafnet/README.md) (ECCV'2022)
- ✅ [Restormer](configs/restormer/README.md) (CVPR'2022)

</details>

Please refer to [model_zoo](https://mmediting.readthedocs.io/en/1.x/3_model_zoo.html) for more details.

## Contributing

We appreciate all contributions to improve MMEditing. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/tree/2.x/CONTRIBUTING.md) in MMCV and [CONTRIBUTING.md](https://github.com/open-mmlab/mmengine/blob/main/CONTRIBUTING.md) in MMEngine for more details about the contributing guideline.

## Acknowledgement

MMEditing is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If MMEditing is helpful to your research, please cite it as below.

```bibtex
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
Please refer to [LICENSES](LICENSE) for the careful check, if you are using our code for commercial matters.

## Projects in OpenMMLab 2.0

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification/tree/1.x): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/3.x): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/1.x): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate/tree/1.x): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/1.x): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr/tree/1.x): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose/tree/1.x): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d/tree/1.x): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup/tree/1.x): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor/tree/1.x): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot/tree/1.x): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2/tree/1.x): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking/tree/1.x): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow/tree/1.x): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting/tree/1.x): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration/tree/1.x): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
