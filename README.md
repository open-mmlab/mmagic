<div align="center">
  <img src="resources/mmediting-logo.png" width="500px"/>
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
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmediting.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![license](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)

[📘Documentation](https://mmediting.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmediting.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmediting.readthedocs.io/en/latest/_tmp/modelzoo.html) |
[🆕Update News](https://github.com/open-mmlab/mmediting/blob/master/docs/en/changelog.md) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmediting/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmediting/issues)

</div>

## Introduction

English | [简体中文](/README_zh-CN.md)

MMEditing is an open-source image and video editing toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Currently, MMEditing support the following tasks:

<div align="center">
  <img src="https://user-images.githubusercontent.com/12756472/158984079-c4754015-c1f6-48c5-ac46-62e79448c372.jpg"/>
</div>

The master branch works with **PyTorch 1.5+**.

Some Demos:

https://user-images.githubusercontent.com/12756472/158972852-be5849aa-846b-41a8-8687-da5dee968ac7.mp4

https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4


### Major features

- **Modular design**

  We decompose the editing framework into different components and one can easily construct a customized editor framework by combining different modules.

- **Support of multiple tasks in editing**

  The toolbox directly supports popular and contemporary *inpainting*, *matting*, *super-resolution* and *generation* tasks.

- **State of the art**

  The toolbox provides state-of-the-art methods in inpainting/matting/super-resolution/generation.

Note that **MMSR** has been merged into this repo, as a part of MMEditing.
With elaborate designs of the new framework and careful implementations,
hope MMEditing could provide better experience.

## News

- [2022-04-01] v0.14.0 was released.
  - Support TOFlow in video frame interpolation
- [2022-03-01] v0.13.0 was released.
  - Support CAIN
  - Support EDVR-L
  - Support running in Windows
- [2022-02-11] Switch to **PyTorch 1.5+**. The compatibility to earlier versions of PyTorch will no longer be guaranteed.

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation

MMEditing depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

**Step 2.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```
pip3 install openmim
mim install mmcv-full
```

**Step 3.**
Install MMEditing from source.

```
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

Please refer to [install.md](../../wiki/1.-Installation) for more detailed instruction.

## Getting Started

Please see [getting_started.md](docs/en/getting_started.md) and [demo.md](docs/en/demo.md) for the basic usage of MMEditing.

## Model Zoo

Supported algorithms:

<details open>
<summary>Inpainting</summary>

- [x] [Global&Local](configs/inpainting/global_local/README.md) (ToG'2017)
- [x] [DeepFillv1](configs/inpainting/deepfillv1/README.md) (CVPR'2018)
- [x] [PConv](configs/inpainting/partial_conv/README.md) (ECCV'2018)
- [x] [DeepFillv2](configs/inpainting/deepfillv2/README.md) (CVPR'2019)

</details>

<details open>
<summary>Matting</summary>

- [x] [DIM](configs/mattors/dim/README.md) (CVPR'2017)
- [x] [IndexNet](configs/mattors/indexnet/README.md) (ICCV'2019)
- [x] [GCA](configs/mattors/gca/README.md) (AAAI'2020)

</details>

<details open>
<summary>Image-Super-Resolution</summary>

- [x] [SRCNN](configs/restorers/srcnn/README.md) (TPAMI'2015)
- [x] [SRResNet&SRGAN](configs/restorers/srresnet_srgan/README.md) (CVPR'2016)
- [x] [EDSR](configs/restorers/edsr/README.md) (CVPR'2017)
- [x] [ESRGAN](configs/restorers/esrgan/README.md) (ECCV'2018)
- [x] [RDN](configs/restorers/rdn/README.md) (CVPR'2018)
- [x] [DIC](configs/restorers/dic/README.md) (CVPR'2020)
- [x] [TTSR](configs/restorers/ttsr/README.md) (CVPR'2020)
- [x] [GLEAN](configs/restorers/glean/README.md) (CVPR'2021)
- [x] [LIIF](configs/restorers/liif/README.md) (CVPR'2021)

</details>

<details open>
<summary>Video-Super-Resolution</summary>

- [x] [EDVR](configs/restorers/edvr/README.md) (CVPR'2019)
- [x] [TOF](configs/restorers/tof/README.md) (IJCV'2019)
- [x] [TDAN](configs/restorers/tdan/README.md) (CVPR'2020)
- [x] [BasicVSR](configs/restorers/basicvsr/README.md) (CVPR'2021)
- [x] [IconVSR](configs/restorers/iconvsr/README.md) (CVPR'2021)
- [x] [BasicVSR++](configs/restorers/basicvsr_plusplus/README.md) (CVPR'2022)
- [x] [RealBasicVSR](configs/restorers/real_basicvsr/README.md) (CVPR'2022)

</details>

<details open>
<summary>Generation</summary>

- [x] [CycleGAN](configs/synthesizers/cyclegan/README.md) (ICCV'2017)
- [x] [pix2pix](configs/synthesizers/pix2pix/README.md) (CVPR'2017)

</details>

<details open>
<summary>Video Interpolation</summary>

- [x] [TOFlow](configs/video_interpolators/tof/README.md) (IJCV'2019)
- [x] [CAIN](configs/video_interpolators/cain/README.md) (AAAI'2020)

</details>

Please refer to [model_zoo](https://mmediting.readthedocs.io/en/latest/_tmp/modelzoo.html) for more details.

## Contributing

We appreciate all contributions to improve MMEditing. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmdetection/blob/master/.github/CONTRIBUTING.md) in MMCV for the contributing guideline.

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

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
