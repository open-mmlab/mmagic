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
[🛠️Installation](https://mmediting.readthedocs.io/en/1.x/get_started.html#installation) |
[👀Model Zoo](https://mmediting.readthedocs.io/en/1.x/model_zoo.html) |
[🆕Update News](docs/en/notes/changelog.md) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmediting/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmediting/issues)

</div>

English | [简体中文](/README_zh-CN.md)

## Introduction

MMEditing is an open-source image and video editing&&generating toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Currently, MMEditing support the following tasks:

<div align="center">
  <img src="https://user-images.githubusercontent.com/22982797/191167628-2ac529d6-6614-4b53-ad65-0cfff909aa7d.jpg"/>
</div>

The master branch works with **PyTorch 1.5+**.

Some Demos:

https://user-images.githubusercontent.com/12756472/158972852-be5849aa-846b-41a8-8687-da5dee968ac7.mp4

https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4

<table>
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

- \[2022-08-31\] 🎉[MMGeneration](https://github.com/open-mmlab/mmgeneration/tree/1.x) was merged into MMEditing! And we are calling for your suggestion!
- \[2022-08-31\] v1.0.0rc0 was released. This release introduced a brand new and flexible training & test engine, but it's still in progress. Welcome
to try according to [the documentation](https://mmediting.readthedocs.io/en/1.x/).
- \[2022-06-01\] v0.15.0 was released.
  - Support FLAVR
  - Support AOT-GAN
  - Support CAIN with ReduceLROnPlateau Scheduler
- \[2022-04-01\] v0.14.0 was released.
  - Support TOFlow in video frame interpolation
- \[2022-03-01\] v0.13.0 was released.
  - Support CAIN
  - Support EDVR-L
  - Support running in Windows
- \[2022-02-11\] Switch to **PyTorch 1.5+**. The compatibility to earlier versions of PyTorch will no longer be guaranteed.

Please refer to [changelog.md](docs/en/notes/changelog.md) for details and release history.

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
git clone -b 1.x shttps://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

Please refer to [get_started.md](docs/en/get_started.md) for more detailed instruction.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) and [inference.md](docs/en/user_guides/inference.md) for the basic usage of MMEditing.

## Model Zoo

Supported algorithms:

<details open>
<summary>Inpainting</summary>

- [x] [Global&Local](configs/global_local/README.md) (ToG'2017)
- [x] [DeepFillv1](configs/deepfillv1/README.md) (CVPR'2018)
- [x] [PConv](configs/partial_conv/README.md) (ECCV'2018)
- [x] [DeepFillv2](configs/deepfillv2/README.md) (CVPR'2019)
- [x] [AOT-GAN](configs/aot_gan/README.md) (TVCG'2021)

</details>

<details open>
<summary>Matting</summary>

- [x] [DIM](configs/dim/README.md) (CVPR'2017)
- [x] [IndexNet](configs/indexnet/README.md) (ICCV'2019)
- [x] [GCA](configs/gca/README.md) (AAAI'2020)

</details>

<details open>
<summary>Image-Super-Resolution</summary>

- [x] [SRCNN](configs/srcnn/README.md) (TPAMI'2015)
- [x] [SRResNet&SRGAN](configs/srgan_resnet/README.md) (CVPR'2016)
- [x] [EDSR](configs/edsr/README.md) (CVPR'2017)
- [x] [ESRGAN](configs/esrgan/README.md) (ECCV'2018)
- [x] [RDN](configs/rdn/README.md) (CVPR'2018)
- [x] [DIC](configs/dic/README.md) (CVPR'2020)
- [x] [TTSR](configs/ttsr/README.md) (CVPR'2020)
- [x] [GLEAN](configs/glean/README.md) (CVPR'2021)
- [x] [LIIF](configs/liif/README.md) (CVPR'2021)
- [x] [Real-ESRGAN](configs/real_esrgan/README.md) (ICCVW'2021)

</details>

<details open>
<summary>Video-Super-Resolution</summary>

- [x] [EDVR](configs/edvr/README.md) (CVPR'2019)
- [x] [TOF](configs/tof/README.md) (IJCV'2019)
- [x] [TDAN](configs/tdan/README.md) (CVPR'2020)
- [x] [BasicVSR](configs/basicvsr/README.md) (CVPR'2021)
- [x] [IconVSR](configs/iconvsr/README.md) (CVPR'2021)
- [x] [BasicVSR++](configs/basicvsr_pp/README.md) (CVPR'2022)
- [x] [RealBasicVSR](configs/real_basicvsr/README.md) (CVPR'2022)

</details>

<details open>
<summary>Video Interpolation</summary>

- [x] [TOFlow](configs/tof/README.md) (IJCV'2019)
- [x] [CAIN](configs/cain/README.md) (AAAI'2020)
- [x] [FLAVR](configs/flavr/README.md) (CVPR'2021)

</details>


<details open>
<summary>Unconditional GANs (click to collapse)</summary>

- ✅ [DCGAN](configs/dcgan/README.md) (ICLR'2016)
- ✅ [WGAN-GP](configs/wgan-gp/README.md) (NIPS'2017)
- ✅ [LSGAN](configs/lsgan/README.md) (ICCV'2017)
- ✅ [GGAN](configs/ggan/README.md) (arXiv'2017)
- ✅ [PGGAN](configs/pggan/README.md) (ICLR'2018)
- ✅ [StyleGANV1](configs/styleganv1/README.md) (CVPR'2019)
- ✅ [StyleGANV2](configs/styleganv2/README.md) (CVPR'2020)
- ✅ [StyleGANV3](configs/styleganv3/README.md) (NeurIPS'2021)

</details>

<details open>
<summary>Conditional GANs (click to collapse)</summary>

- ✅ [SNGAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [Projection GAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [SAGAN](configs/sagan/README.md) (ICML'2019)
- ✅ [BIGGAN/BIGGAN-DEEP](configs/biggan/README.md) (ICLR'2019)

</details>

<details open>
<summary>Image2Image Translation (click to collapse)</summary>

- ✅ [Pix2Pix](configs/pix2pix/README.md) (CVPR'2017)
- ✅ [CycleGAN](configs/cyclegan/README.md) (ICCV'2017)

</details>

<details open>
<summary>Internal Learning (click to collapse)</summary>

- ✅ [SinGAN](configs/singan/README.md) (ICCV'2019)

</details>

Please refer to [model_zoo](https://mmediting.readthedocs.io/en/1.x/model_zoo.html) for more details.

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

This project is released under the [Apache 2.0 license](LICENSE). Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

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
