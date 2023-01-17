<div id="top" align="center">
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

[Documentation](https://mmediting.readthedocs.io/en/1.x/) |
[Installation](https://mmediting.readthedocs.io/en/1.x/2_get_started.html#installation) |
[Model Zoo](https://mmediting.readthedocs.io/en/1.x/3_model_zoo.html) |
[Update News](docs/en/changelog.md) |
[Ongoing Projects](https://github.com/open-mmlab/mmediting/projects) |
[Reporting Issues](https://github.com/open-mmlab/mmediting/issues)

English | [简体中文](README_zh-CN.md)

</div>

## Table of Contents
- [Introduction](#introduction)
- [What's New](#whats-new)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Model Zoo](#model-zoo)
- [Demos](#demos)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)
- [Projects in OpenMMLab 2.0](#projects-in-openmmlab-20)


## Introduction

MMEditing is an open-source image and video editing&generating toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Currently, MMEditing support the following tasks:

<table align="center">
<thead>
  <tr>
    <td>
<div >
  <b>Text-to-Image</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212800234-5dfcecc2-5ef4-4c08-a55b-aa020bf3ff53.png" width="220" height="220"/>
</div></td>
    <td>
<div>
  <b>3D-aware Generation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212802550-ed80c1da-ac46-484e-b182-caeb240ea48f.png" width="220" height="220"/>
</div></td>
    <td>
<div >
  <b>Image Restoration</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212806941-fcb775da-876a-4020-aa04-07f96e5c50ea.png" width="220" height="220"/>
</div></td>
  </tr>
</thead>
<thead>
  <tr>
    <td>
<div >
  <b>Image Generation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212686396-1b778277-7acc-4e48-ad77-e89f3a9cc09a.png" width="220" height="220"/>
</div></td>
    <td>
<div >
  <b>Inpainting</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212686578-3c47b5c2-c74e-4be2-88c6-fa486f0f28de.png" width="220" height="220"/>
</div></td>
    <td>
<div>
  <b>Frame Interpolation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212685664-2d2cbed4-74c0-4d9f-90d8-3beefcb51918.png" width="220" height="220"/>
</div></td>
  </tr>
</thead>
<thead>
  <tr>
    <td>
<div>
  <b>Matting</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212798790-dacfb899-ee3f-4c3c-8cd9-44823832415d.png" width="220" height="220"/>
</div></td>
    <td>
<div >
  <b>Image Translation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212804088-c3711a50-1577-4dd6-97a4-cd3c3d37e189.png" width="220" height="220"/>
</div></td>
    <td>
<div >
  <b>Super Resolution</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12782558/212799468-34085fbf-d38e-41ab-ad5d-bbf54f5088e9.png" width="220" height="220"/>
</div></td>
  </tr>
</thead>
</table>

And more...

The best practice on our main 1.x branch works with **Python 3.8+** and **PyTorch 1.9+**.

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

<p align="right"><a href="#top">Back to top</a></p>

## What's New <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

### New release [**MMEditing v1.0.0rc5**](https://github.com/open-mmlab/mmediting/releases/tag/v1.0.0rc5) [01/2023]
  A brand new version of **MMEditing v1.0.0rc5** was released in 04/01/2023:
- Support well-known text-to-image method [Stable Diffusion](configs/stable_diffusion/README.md)!
- Support an efficient image restoration algorithm [Restormer](configs/restormer/README.md)!
- Support a new text-to-image algorithm [GLIDE](projects/glide/configs/README.md)!
- Support swin based image restoration algorithm [SwinIR](configs/swinir/README.md)!
- [Projects](projects/README.md) is opened for community to add projects to MMEditing.

### New release [**MMEditing v1.0.0rc4**](https://github.com/open-mmlab/mmediting/releases/tag/v1.0.0rc4) [12/2022]
  A brand new version of **MMEditing v1.0.0rc4** was released in 05/12/2022:
- Support Text2Image Task! [Disco-Diffusion](configs/disco_diffusion/README.md)
- Support 3D-aware Generation Task! [EG3D](configs/eg3d/README.md)
- Support image colorization.

### New features
- Support all the tasks, models, metrics, and losses in [MMGeneration](https://github.com/open-mmlab/mmgeneration) 😍.
- Unifies interfaces of all components based on [MMEngine](https://github.com/open-mmlab/mmengine).
- Support patch-based and slider-based image and video comparison viewer.

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

<p align="right"><a href="#top">Back to top</a></p>

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

<p align="right"><a href="#top">Back to top</a></p>

## Getting Started

Please see [quick run](docs/en/get_started/quick_run.md) and [inference](docs/en/user_guides/inference.md) for the basic usage of MMEditing.

<p align="right"><a href="#top">Back to top</a></p>

## Model Zoo
<div align="center">
  <b>Supported algorithms</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Conditional GANs</b>
      </td>
      <td>
        <b>Unconditional GANs</b>
      </td>
      <td>
        <b>Image Restoration</b>
      </td>
      <td>
        <b>Image Super-Resolution</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/sngan_proj/README.md">SNGAN (ICLR'2018)</a></li>
            <li><a href="configs/sngan_proj/README.md">Projection GAN (ICLR'2018)</a></li>
            <li><a href="configs/sagan/README.md">SAGAN (ICML'2019)</a></li>
            <li><a href="configs/biggan/README.md">BIGGAN/BIGGAN-DEEP (ICLR'2018)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/dcgan/README.md">DCGAN (ICLR'2016)</a></li>
          <li><a href="configs/wgan-gp/README.md">WGAN-GP (NeurIPS'2017)</a></li>
          <li><a href="configs/lsgan/README.md">LSGAN (ICCV'2017)</a></li>
          <li><a href="configs/ggan/README.md">PGGAN (ArXiv'2017)</a></li>
          <li><a href="configs/pggan/README.md">PGGAN (ICLR'2018)</a></li>
          <li><a href="configs/singan/README.md">SinGAN (ICCV'2019)</a></li>
          <li><a href="configs/styleganv1/README.md">StyleGANV1 (CVPR'2019)</a></li>
          <li><a href="configs/styleganv2/README.md">StyleGANV2 (CVPR'2019)</a></li>
          <li><a href="configs/styleganv3/README.md">StyleGANV3 (NeurIPS'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/swinir/README.md">SwinIR (ICCVW'2021)</a></li>
          <li><a href="configs/nafnet/README.md">NAFNet (ECCV'2022)</a></li>
          <li><a href="configs/restormer/README.md">Restormer (CVPR'2022)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/srcnn/README.md">SRCNN (TPAMI'2015)</a></li>
          <li><a href="configs/srgan_resnet/README.md">SRResNet&SRGAN (CVPR'2016)</a></li>
          <li><a href="configs/edsr/README.md">EDSR (CVPR'2017)</a></li>
          <li><a href="configs/esrgan/README.md">ESRGAN (ECCV'2018)</a></li>
          <li><a href="configs/rdn/README.md">RDN (CVPR'2018)</a></li>
          <li><a href="configs/dic/README.md">DIC (CVPR'2020)</a></li>
          <li><a href="configs/ttsr/README.md">TTSR (CVPR'2020)</a></li>
          <li><a href="configs/glean/README.md">GLEAN (CVPR'2021)</a></li>
          <li><a href="configs/liif/README.md">LIIF (CVPR'2021)</a></li>
          <li><a href="configs/real_esrgan/README.md">Real-ESRGAN (ICCVW'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
<tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Video Super-Resolution</b>
      </td>
      <td>
        <b>Video Interpolation</b>
      </td>
      <td>
        <b>Image Colorization</b>
      </td>
      <td>
        <b>Image Translation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/edvr/README.md">EDVR (CVPR'2018)</a></li>
            <li><a href="configs/tof/README.md">TOF (IJCV'2019)</a></li>
            <li><a href="configs/tdan/README.md">TDAN (CVPR'2020)</a></li>
            <li><a href="configs/basicvsr/README.md">BasicVSR (CVPR'2021)</a></li>
            <li><a href="configs/iconvsr/README.md">IconVSR (CVPR'2021)</a></li>
            <li><a href="configs/basicvsr_pp/README.md">BasicVSR++ (CVPR'2022)</a></li>
            <li><a href="configs/real_basicvsr/README.md">RealBasicVSR (CVPR'2022)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/tof/README.md">TOFlow (IJCV'2019)</a></li>
          <li><a href="configs/cain/README.md">CAIN (AAAI'2020)</a></li>
          <li><a href="configs/flavr/README.md">FLAVR (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/inst_colorization/README.md">DIM (CVPR'2020)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/pix2pix/README.md">Pix2Pix (CVPR'2017)</a></li>
          <li><a href="configs/cyclegan/README.md">CycleGAN (ICCV'2017)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
<tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Inpainting</b>
      </td>
      <td>
        <b>Matting</b>
      </td>
      <td>
        <b>Text-to-Image</b>
      </td>
      <td>
        <b>3D-aware Generation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="configs/global_local/README.md">Global&Local (ToG'2017)</a></li>
          <li><a href="configs/deepfillv1/README.md">DeepFillv1 (CVPR'2018)</a></li>
          <li><a href="configs/partial_conv/README.md">PConv (ECCV'2018)</a></li>
          <li><a href="configs/deepfillv2/README.md">DeepFillv2 (CVPR'2019)</a></li>
          <li><a href="configs/aot_gan/README.md">AOT-GAN (TVCG'2019)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/dim/README.md">DIM (CVPR'2017)</a></li>
          <li><a href="configs/indexnet/README.md">IndexNet (ICCV'2019)</a></li>
          <li><a href="configs/mask2former">GCA (AAAI'2020)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="projects/glide/configs/README.md">GLIDE (NeurIPS'2021)</a></li>
          <li><a href="configs/disco_diffusion/README.md">Disco-Diffusion (2022)</a></li>
          <li><a href="configs/stable_diffusion/README.md">Stable-Diffusion (2022)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/eg3d/README.md">EG3D (CVPR'2022)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

Please refer to [model_zoo](https://mmediting.readthedocs.io/en/1.x/model_zoo/overview.html) for more details.

<p align="right"><a href="#top">Back to top</a></p>

## Demos:

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

<p align="right"><a href="#top">Back to top</a></p>

## Contributing

We appreciate all contributions to improve MMEditing. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/tree/2.x/CONTRIBUTING.md) in MMCV and [CONTRIBUTING.md](https://github.com/open-mmlab/mmengine/blob/main/CONTRIBUTING.md) in MMEngine for more details about the contributing guideline.

<p align="right"><a href="#top">Back to top</a></p>

## Acknowledgement

MMEditing is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

<p align="right"><a href="#top">Back to top</a></p>

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

<p align="right"><a href="#top">Back to top</a></p>

## License

This project is released under the [Apache 2.0 license](LICENSE).
Please refer to [LICENSES](LICENSE) for the careful check, if you are using our code for commercial matters.

<p align="right"><a href="#top">Back to top</a></p>

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

<p align="right"><a href="#top">Back to top</a></p>
