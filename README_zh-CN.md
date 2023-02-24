<div align="center">
  <img src="docs/zh_cn/_static/resources/mmediting-logo.png" width="500px"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
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

[📘使用文档](https://mmediting.readthedocs.io/zh_CN/latest/) |
[🛠️安装教程](https://mmediting.readthedocs.io/zh_CN/latest/install.html) |
[👀模型库](https://mmediting.readthedocs.io/zh_CN/latest/modelzoo.html) |
[🆕更新记录](https://github.com/open-mmlab/mmediting/blob/master/docs/zh_cn/changelog.md) |
[🚀进行中的项目](https://github.com/open-mmlab/mmediting/projects) |
[🤔提出问题](https://github.com/open-mmlab/mmediting/issues)

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## Introduction

MMEditing 是基于 PyTorch 的图像&视频编辑开源工具箱。是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一。

目前 MMEditing 支持下列任务：

<div align="center">
  <img src="https://user-images.githubusercontent.com/12756472/158984079-c4754015-c1f6-48c5-ac46-62e79448c372.jpg"/>
</div>

主分支代码目前支持 **PyTorch 1.5 以上**的版本。

一些示例:

https://user-images.githubusercontent.com/12756472/158972852-be5849aa-846b-41a8-8687-da5dee968ac7.mp4

https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4

### 主要特性

- **模块化设计**

  MMEditing 将编辑框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。

- **支持多种编辑任务**

  MMEditing 支持*修复*、*抠图*、*超分辨率*、*生成*等多种主流编辑任务。

- **SOTA**

  MMEditing 提供修复/抠图/超分辨率/生成等任务最先进的算法。

需要注意的是 **MMSR** 已作为 MMEditing 的一部分并入本仓库。
MMEditing 缜密地设计新的框架并将其精心实现，希望能够为您带来更好的体验。

## 最新进展

MMEditing 同时维护 0.x 和 1.x 版本，详情见[分支维护计划](README_zh-CN.md#分支维护计划)

### 💎 稳定版本

最新的 **0.16.1** 版本已经在 24/02/2023 发布：

- 支持新评价指标 FID 和 KID。
- 支持 ResidualBlockNoBN 模块设置 `groups` 参数。
- 修复 RealESRGAN 测试数据集配置。
- 修复 `pixel-unshuffle` 模块动态输入导出 ONNX 的Bug。

如果像了解更多版本更新细节和历史信息，请阅读[更新日志](/docs/zh_cn/changelog.md)。

### 🌟 1.x 预览版本

全新的 [**MMEditing v1.0.0rc6**](https://github.com/open-mmlab/mmediting/releases/tag/v1.0.0rc6) 已经在 24/02/2023 发布:

- 支持[MMGeneration](https://github.com/open-mmlab/mmgeneration)中的全量任务、模型、优化函数和评价指标 😍。
- 基于[MMEngine](https://github.com/open-mmlab/mmengine)统一了各组件接口。
- 重构之后更加灵活 [architecture](https://mmediting.readthedocs.io/en/1.x/1_overview.html)。
- 支持了著名的文本生成图像方法 [Stable Diffusion](https://github.com/open-mmlab/mmediting/tree/1.x/configs/stable_diffusion/README.md)!
- 支持了一个新的文本到图像生成算法 [GLIDE](https://github.com/open-mmlab/mmediting/tree/1.x/projects/glide/configs/README.md)!
- 支持了一个新的文本到图像生成算法 [Disco-Diffusion](https://github.com/open-mmlab/mmediting/tree/1.x/configs/disco_diffusion/README.md)!
- 支持了3D生成算法 [EG3D](https://github.com/open-mmlab/mmediting/tree/1.x/configs/eg3d/README.md)!
- 支持了一个高效的图像复原算法 [Restormer](https://github.com/open-mmlab/mmediting/tree/1.x/configs/restormer/README.md)!
- 支持了基于swin的图像复原算法 [SwinIR](https://github.com/open-mmlab/mmediting/tree/1.x/configs/swinir/README.md)!
- 支持图像上色算法 [Image Colorization](https://github.com/open-mmlab/mmediting/tree/1.x/configs/inst_colorization/README.md).
- 开启了[Projects](https://github.com/open-mmlab/mmediting/tree/1.x/projects/README.md)以便社区用户添加新的项目到MMEditing.
- 支持了 High-level apis and inferencer.
- 支持Inpainting任务的Gradio交互GUI.
- 支持基于patch以及滑动条的图像和视频可视化比较工具.

在[1.x 分支](https://github.com/open-mmlab/mmediting/tree/1.x)中发现更多特性！欢迎提 Issues 和 PRs！

## 安装

MMEditing 依赖 [PyTorch](https://pytorch.org/) 和 [MMCV](https://github.com/open-mmlab/mmcv)，以下是安装的简要步骤。

**步骤 1.**
依照[官方教程](https://pytorch.org/get-started/locally/)安装PyTorch

**步骤 2.**
使用 [MIM](https://github.com/open-mmlab/mim) 安装 MMCV

```
pip3 install openmim
mim install mmcv-full
```

**步骤 3.**
从源码安装 MMEditing

```
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

更详细的安装指南请参考 [install.md](../../wiki/1.-Installation) 。

## 开始使用

请参考[使用教程](docs/zh_cn/getting_started.md)和[功能演示](docs/zh_cn/demo.md)获取MMEditing的基本用法。

## 模型库

支持的算法:

<details open>
<summary>图像修复</summary>

- [x] [Global&Local](configs/inpainting/global_local/README.md) (ToG'2017)
- [x] [DeepFillv1](configs/inpainting/deepfillv1/README.md) (CVPR'2018)
- [x] [PConv](configs/inpainting/partial_conv/README.md) (ECCV'2018)
- [x] [DeepFillv2](configs/inpainting/deepfillv2/README.md) (CVPR'2019)
- [x] [AOT-GAN](configs/inpainting/AOT-GAN/README.md) (TVCG'2021)

</details>

<details open>
<summary>图像抠图</summary>

- [x] [DIM](configs/mattors/dim/README.md) (CVPR'2017)
- [x] [IndexNet](configs/mattors/indexnet/README.md) (ICCV'2019)
- [x] [GCA](configs/mattors/gca/README.md) (AAAI'2020)

</details>

<details open>
<summary>图像超分辨率</summary>

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
<summary>视频超分辨率</summary>

- [x] [EDVR](configs/restorers/edvr/README.md) (CVPR'2019)
- [x] [TOF](configs/restorers/tof/README.md) (IJCV'2019)
- [x] [TDAN](configs/restorers/tdan/README.md) (CVPR'2020)
- [x] [BasicVSR](configs/restorers/basicvsr/README.md) (CVPR'2021)
- [x] [IconVSR](configs/restorers/iconvsr/README.md) (CVPR'2021)
- [x] [BasicVSR++](configs/restorers/basicvsr_plusplus/README.md) (CVPR'2022)
- [x] [RealBasicVSR](configs/restorers/real_basicvsr/README.md) (CVPR'2022)

</details>

<details open>
<summary>图像生成</summary>

- [x] [CycleGAN](configs/synthesizers/cyclegan/README.md) (ICCV'2017)
- [x] [pix2pix](configs/synthesizers/pix2pix/README.md) (CVPR'2017)

</details>

<details open>
<summary>视频插帧</summary>

- [x] [TOFlow](configs/video_interpolators/tof/README.md) (IJCV'2019)
- [x] [CAIN](configs/video_interpolators/cain/README.md) (AAAI'2020)
- [x] [FLAVR](configs/video_interpolators/flavr/README.md) (CVPR'2021)

</details>

请参考[模型库](https://mmediting.readthedocs.io/en/latest/_tmp/modelzoo.html)了解详情。

## 参与贡献

感谢您为改善 MMEditing 所做的所有贡献。请参阅 MMCV 中的 [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) 以获取贡献指南。

## 致谢

MMEditing 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 分支维护计划

MMCV 目前有两个分支，分别是 master 和 2.x 分支，它们会经历以下三个阶段：

| 阶段   | 时间                  | 分支                                                         | 说明                                                                                                     |
| ------ | --------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| 公测期 | 2022/9/1 - 2022.12.31 | 公测版代码发布在 1.x 分支；默认主分支 master 仍对应 0.x 版本 | master 和 1.x 分支正常进行迭代                                                                           |
| 兼容期 | 2023/1/1 - 2023.12.31 | **切换默认主分支 master 为 1.x 版本**；0.x 分支对应 0.x 版本 | 保持对旧版本 0.x 的维护和开发，响应用户需求，但尽量不引进破坏旧版本兼容性的改动；master 分支正常进行迭代 |
| 维护期 | 2024/1/1 - 待定       | 默认主分支 master 为 1.x 版本；0.x 分支对应 0.x 版本         | 0.x 分支进入维护阶段，不再进行新功能支持；master 分支正常进行迭代                                        |

## 引用

如果 MMEditing 对您的研究有所帮助，请按照如下 bibtex 引用它。

```bibtex
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```

## 许可证

本项目开源自 [Apache 2.0 license](LICENSE)。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMEval](https://github.com/open-mmlab/mmeval): 统一开放的跨框架算法评测库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=K0QI8ByU)，或通过群主小喵加入微信官方交流群。

<div align="center">
<img src="docs/zh_cn/_static/image/zhihu_qrcode.jpg" height="500" />  <img src="https://user-images.githubusercontent.com/25839884/203927852-e15def4d-a0eb-4dfc-9bfb-7cf09ea945d0.png" height="500" /> <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/wechat_qrcode.jpg" height="500" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
