<div align="center">
  <img src="resources/mmediting-logo.png" width="500px"/>
</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmediting.readthedocs.io/en/latest/)
[![build](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![pypi](https://img.shields.io/pypi/v/mmediting)](https://pypi.org/project/mmediting)
[![license](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/master/LICENSE)

## Introduction

The master branch works with **PyTorch 1.3 to 1.5**.

MMEditing is an open source image and video editing toolbox based on PyTorch. It is a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

<div align="left">
  <img src="resources/mmediting-demo.jpg"/>
</div>

### Major features

- **Modular design**

  We decompose the editing framework into different components and one can easily construct a customized editor framework by combining different modules.

- **Support of multiple tasks in editing**

  The toolbox directly supports popular and contemporary *inpainting*, *matting*, *super-resolution* ang *generation* tasks.

- **State of the art**

  The toolbox provides state-of-the-art methods in inpainting/matting/super-resolution/generation.


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.5 was released in 09/07/2020.

## Benchmark and model zoo

Please refer to [model_zoo.rst](docs/model_zoo.rst) for more details.

## Installation

Please refer to [install.md](docs/install.md) for installation.


## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMEditing.

## Contributing

We appreciate all contributions to improve MMEditing. Please refer to [CONTRIBUTING.md in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/.github/CONTRIBUTING.md) for the contributing guideline.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{mmediting2020,
  author =       {Jiamin He, Rui Xu, Xintao Wang, Liming Jiang, Wayne Wu, Chen Qian, Kai Chen, Dahua Lin and Chen Change Loy},
  title =        {mmediting},
  howpublished = {\url{https://github.com/open-mmlab/mmediting}},
  year =         {2020}
}
```

## Contact

This repo is currently maintained by Jiamin He ([@hejm37](https://github.com/hejm37)), Rui Xu ([@nbei](https://github.com/nbei)), Xintao Wang ([@xinntao](https://github.com/xinntao)), Liming Jiang([@EndlessSora](https://github.com/EndlessSora)).
