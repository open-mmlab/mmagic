# MMSR

MMSR is an open source image and video super-resolution toolbox based on PyTorch. It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/). MMSR is based on our previous projects: [BasicSR](https://github.com/xinntao/BasicSR), [ESRGAN](https://github.com/xinntao/ESRGAN), and [EDVR](https://github.com/xinntao/EDVR).

### Highlights
- **A unified framework** suitable for image and video super-resolution tasks. It is also easy to adapt to other restoration tasks, e.g., deblurring, denoising, etc.
- **State of the art**: It includes several winning methods in competitions: such as ESRGAN (PIRM18), EDVR (NTIRE19).
- **Easy to extend**: It is easy to try new research ideas based on the code base.


### Updates
[2019-07-25] MMSR v0.1 is released.

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [mmdetection](https://github.com/open-mmlab/mmdetection)'s dcn implementation. Please first compile it.
  ```
  cd ./codes/models/archs/dcn
  python setup.py develop
  ```
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard:
  - PyTorch >= 1.1: `pip install tb-nightly future`


## Dataset Preparation
We use datasets in LDMB format for faster IO speed. Please refer to [DATASETS.md](datasets/DATASETS.md) for more details.

## Training and Testing
Please see [wiki- Training and Testing](https://github.com/open-mmlab/mmsr/wiki/Training-and-Testing) for the basic usage, *i.e.,* training and testing.

## Model Zoo and Baselines
Results and pre-trained models are available in the [wiki-Model Zoo](https://github.com/open-mmlab/mmsr/wiki/Model-Zoo).

## Contributing
We appreciate all contributions. Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/CONTRIBUTING.md) for contributing guideline.

**Python code style**<br/>
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style. We use [flake8](http://flake8.pycqa.org/en/latest/) as the linter and [yapf](https://github.com/google/yapf) as the formatter. Please upgrade to the latest yapf (>=0.27.0) and refer to the [yapf configuration](.style.yapf) and [flake8 configuration](.flake8).

> Before you create a PR, make sure that your code lints and is formatted by yapf.

## License
This project is released under the Apache 2.0 license.
