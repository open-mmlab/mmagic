<div align="center">
  <img src="resources/mmediting-logo.png" width="500px"/>
</div>

## Introduction

[English](/README.md) | 简体中文


[![Documentation](https://readthedocs.org/projects/mmediting/badge/?version=latest)](https://mmediting.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![PyPI](https://badge.fury.io/py/mmedit.svg)](https://pypi.org/project/mmedit/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)

MMEditing 是基于 PyTorch 的图像&视频编辑开源工具箱。是 [OpenMMLab](https://open-mmlab.github.io/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.3+**。
请注意，MMEditing 将在2021年10月后仅支持 **PyTorch 1.5 以上**的版本，并不再保证对较早版本的兼容性。

文献资料: https://mmediting.readthedocs.io/en/latest/.

<div align="left">
  <img src="resources/mmediting-demo.jpg"/>
</div>

### 主要特性

- **模块化设计**

  MMEditing 将编辑框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。

- **支持多种编辑任务**

  MMEditing 支持*修复*、*抠图*、*超分辨率*、*生成*等多种主流编辑任务。

- **SOTA**

  MMEditing 提供修复/抠图/超分辨率/生成等任务最先进的算法.

## 模型库

支持的算法:

<details open>
<summary>修复</summary>

- [x] [DeepFillv1](configs/inpainting/deepfillv1/README.md) (CVPR'2018)
- [x] [DeepFillv2](configs/inpainting/deepfillv2/README.md) (CVPR'2019)
- [x] [Global&Local](configs/inpainting/global_local/README.md) (ToG'2017)
- [x] [PConv](configs/inpainting/partial_conv/README.md) (ECCV'2018)

</details>

<details open>
<summary>抠图</summary>

- [x] [DIM](configs/mattors/dim/README.md) (CVPR'2017)
- [x] [GCA](configs/mattors/gca/README.md) (AAAI'2020)
- [x] [IndexNet](configs/mattors/indexnet/README.md) (ICCV'2019)

</details>

<details open>
<summary>超分辨率</summary>

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
<summary>生成</summary>

- [x] [CycleGAN](configs/synthesizers/cyclegan/README.md) (ICCV'2017)
- [x] [pix2pix](configs/synthesizers/pix2pix/README.md) (CVPR'2017)

</details>

请参考[模型库](https://mmediting.readthedocs.io/en/latest/modelzoo.html)了解详情。
## 许可证

本项目开源自 [Apache 2.0 license](LICENSE)。

## 更新记录

v0.10.0 版本已于 2021 年 8 月 12 日发布.

需要注意的是 **MMSR** 已作为 MMEditing 的一部分并入本仓库。
MMEditing 缜密地设计新的框架并将其精心实现，希望能够为您带来更好的体验。


## 安装

请参考[安装指南](docs/install.md)进行安装。

## 开始使用

请参考[使用教程](docs/getting_started.md)获取MMEditing的基本用法。



## 引用

如果您觉得 MMEditing 对您的研究有所帮助，请考虑引用它：

```bibtex
@misc{mmediting2020,
    title={OpenMMLab Editing Estimation Toolbox and Benchmark},
    author={MMEditing Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year={2020}
}
```

## 参与贡献

感谢您为改善 MMEditing 所做的所有贡献。请参阅 [CONTRIBUTING.md in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/.github/CONTRIBUTING.md) 以获取贡献准则。

## 致谢

MMEditing 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 生成模型工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=GJP18SjI)

<div align="center">
<img src="resources/zhihu_qrcode.jpg" height="400" />  <img src="resources/qq_group2_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
