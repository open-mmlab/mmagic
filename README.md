<div align="center">
  <img src="resources/mmediting-logo.png" width="500px"/>
</div>

## Introduction

English | [简体中文](/README_zh-CN.md)

[![Documentation](https://readthedocs.org/projects/mmediting/badge/?version=latest)](https://mmediting.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![PyPI](https://badge.fury.io/py/mmedit.svg)](https://pypi.org/project/mmedit/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)


MMEditing is an open source image and video editing toolbox based on PyTorch. It is a part of the [OpenMMLab](https://open-mmlab.github.io/) project.

The master branch works with **PyTorch 1.3+**.
Please kindly note that MMEditing will switch to **PyTorch 1.5+** from Oct. 2021. The compatibility to earlier versions of PyTorch will no longer be guaranteed.

Documentation: https://mmediting.readthedocs.io/en/latest/.

<div align="left">
  <img src="resources/mmediting-demo.jpg"/>
</div>

### Major features

- **Modular design**

  We decompose the editing framework into different components and one can easily construct a customized editor framework by combining different modules.

- **Support of multiple tasks in editing**

  The toolbox directly supports popular and contemporary *inpainting*, *matting*, *super-resolution* and *generation* tasks.

- **State of the art**

  The toolbox provides state-of-the-art methods in inpainting/matting/super-resolution/generation.

## Model Zoo

Supported algorithms:

<details open>
<summary>Inpainting</summary>

- [x] [DeepFillv1](configs/inpainting/deepfillv1/README.md) (CVPR'2018)
- [x] [DeepFillv2](configs/inpainting/deepfillv2/README.md) (CVPR'2019)
- [x] [Global&Local](configs/inpainting/global_local/README.md) (ToG'2017)
- [x] [PConv](configs/inpainting/partial_conv/README.md) (ECCV'2018)

</details>

<details open>
<summary>Matting</summary>

- [x] [DIM](configs/mattors/dim/README.md) (CVPR'2017)
- [x] [GCA](configs/mattors/gca/README.md) (AAAI'2020)
- [x] [IndexNet](configs/mattors/indexnet/README.md) (ICCV'2019)

</details>

<details open>
<summary>Super-Resolution</summary>

- [x] [BasicVSR](configs/restorers/basicvsr/README.md) (CVPR'2021)
- [x] [BasicVSR++](configs/restorers/basicvsr_plusplus/README.md) (NTIRE'2021)
- [x] [EDSR](configs/restorers/edsr/README.md) (CVPR'2017)
- [x] [EDVR](configs/restorers/edvr/README.md) (CVPR'2019)
- [x] [ESRGAN](configs/restorers/esrgan/README.md) (ECCV'2018)
- [x] [GLEAN](configs/restorers/glean/README.md) (CVPR'2021)
- [x] [IconVSR](configs/restorers/iconvsr/README.md) (CVPR'2021)
- [x] [LIIF](configs/restorers/liif/README.md) (CVPR'2021)
- [x] [RDN](configs/restorers/rdn/README.md) (CVPR'2018)
- [x] [SRCNN](configs/restorers/srcnn/README.md) (TPAMI'2015)
- [x] [SRResNet&SRGAN](configs/restorers/srresnet_srgan/README.md) (CVPR'2016)
- [x] [TDAN](configs/restorers/tdan/README.md) (CVPR'2020)
- [x] [TOF](configs/restorers/tof/README.md) (IJCV'2019)
- [x] [TTSR](configs/restorers/ttsr/README.md) (CVPR'2020)
- [x] [DIC](configs/restorers/dic/README.md) (CVPR'2020)

</details>

<details open>
<summary>Generation</summary>

- [x] [CycleGAN](configs/synthesizers/cyclegan/README.md) (ICCV'2017)
- [x] [pix2pix](configs/synthesizers/pix2pix/README.md) (CVPR'2017)

</details>


Please refer to [model_zoo](https://mmediting.readthedocs.io/en/latest/modelzoo.html) for more details.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.10.0 was released in 2021-8-12.

Note that **MMSR** has been merged into this repo, as a part of MMEditing.
With elaborate designs of the new framework and careful implementations,
hope MMEditing could provide better experience.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMEditing.



## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmediting2020,
    title={OpenMMLab Editing Estimation Toolbox and Benchmark},
    author={MMEditing Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year={2020}
}
```


## Contributing

We appreciate all contributions to improve MMEditing. Please refer to [CONTRIBUTING.md in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMEditing is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): A powerful toolkit for generative models.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
