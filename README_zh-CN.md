<div align="center">
  <img src="docs/zh_cn/_static/image/mmediting-logo.png" width="500px"/>
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
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmediting.readthedocs.io/en/1.x/)
[![badge](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![license](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/1.x/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)

[📘使用文档](https://mmediting.readthedocs.io/zh_CN/1.x/) |
[🛠️安装教程](https://mmediting.readthedocs.io/zh_CN/1.x/2_get_started.htmll) |
[👀模型库](https://mmediting.readthedocs.io/zh_CN/1.x/3_model_zoo.html) |
[🆕更新记录](docs/zh_cn/community/changelog.md) |
[🚀进行中的项目](https://github.com/open-mmlab/mmediting/projects) |
[🤔提出问题](https://github.com/open-mmlab/mmediting/issues)

[English](README.md) | 简体中文

</div>

## 介绍

MMEditing 是基于 PyTorch 的图像&视频编辑和生成开源工具箱。是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一。

目前 MMEditing 支持下列任务：

<div align="center">
  <img src="https://user-images.githubusercontent.com/22982797/191167628-2ac529d6-6614-4b53-ad65-0cfff909aa7d.jpg"/>
</div>

主分支代码目前支持 **PyTorch 1.5 以上**的版本。

一些示例:

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

### 主要特性

- **模块化设计**

  MMEditing 将编辑框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。

- **支持多种任务**

  MMEditing 支持*修复*、*抠图*、*超分辨率*、*插帧*等多种主流编辑任务。

- **高效的分布式训练**

  得益于 [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), 动态模型的训练可以轻松实现。

- **SOTA**

  MMEditing 提供修复/抠图/超分辨率/插帧/生成等任务最先进的算法。

需要注意的是 **MMSR** 已作为 MMEditing 的一部分并入本仓库。
MMEditing 缜密地设计新的框架并将其精心实现，希望能够为您带来更好的体验。

## 最新进展

### 🌟 1.x 预览版本

全新的 [**MMEditing v1.0.0rc4**](https://github.com/open-mmlab/mmediting/releases/tag/v1.0.0rc4) 已经在 05/12/2022 发布:

- 支持了图文生成任务! [Disco-Diffusion](configs/disco_diffusion/README.md)
- 支持了3D级图像生成任务! [EG3D](configs/eg3d/README.md)
- 支持[MMGeneration](https://github.com/open-mmlab/mmgeneration)中的全量任务、模型、优化函数和评价指标 😍。
- 基于[MMEngine](https://github.com/open-mmlab/mmengine)统一了各组件接口。
- 支持基于图像子块以及滑动条的图像和视频比较可视化工具。
- 支持图像上色任务。

在[1.x 分支](https://github.com/open-mmlab/mmediting/tree/1.x)中发现更多特性！欢迎提 Issues 和 PRs！

### 💎 稳定版本

最新的 **0.16.0** 版本已经在 31/10/2022 发布：

- `VisualizationHook` 将被启用，建议用户使用 `MMEditVisualizationHook`。
- 修复 FLAVR 的注册问题。
- 修正 RDB 模型中的通道数。

如果像了解更多版本更新细节和历史信息，请阅读[更新日志](docs/en/community/changelog.md)。

## 安装

MMEditing 依赖 [PyTorch](https://pytorch.org/)，[MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)，以下是安装的简要步骤。

**步骤 1.**
依照[官方教程](https://pytorch.org/get-started/locally/)安装PyTorch

**步骤 2.**
使用 [MIM](https://github.com/open-mmlab/mim) 安装 MMCV

```
pip3 install openmim
# wait for more pre-compiled pkgs to release
mim install 'mmcv>=2.0.0rc1'
```

**步骤 3.**
从源码安装 MMEditing

```
git clone -b 1.x https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

更详细的安装指南请参考 [get_started.md](docs/zh_cn/2_get_started.md) 。

## 开始使用

请参考[使用教程](docs/zh_cn/2_get_started.md)和[功能演示](docs/zh_cn/user_guides/3_inference.md)获取MMEditing的基本用法。

## 模型库

支持的算法:

<details open>
<summary>图像修复</summary>

- ✅ [Global&Local](configs/global_local/README.md) (ToG'2017)
- ✅ [DeepFillv1](configs/deepfillv1/README.md) (CVPR'2018)
- ✅ [PConv](configs/partial_conv/README.md) (ECCV'2018)
- ✅ [DeepFillv2](configs/deepfillv2/README.md) (CVPR'2019)
- ✅ [AOT-GAN](configs/aot_gan/README.md) (TVCG'2021)

</details>

<details open>
<summary>图像抠图</summary>

- ✅ [DIM](configs/dim/README.md) (CVPR'2017)
- ✅ [IndexNet](configs/indexnet/README.md) (ICCV'2019)
- ✅ [GCA](configs/gca/README.md) (AAAI'2020)

</details>

<details open>
<summary>图像超分辨率</summary>

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
<summary>视频超分辨率</summary>

- ✅ [EDVR](configs/edvr/README.md) (CVPR'2019)
- ✅ [TOF](configs/tof/README.md) (IJCV'2019)
- ✅ [TDAN](configs/tdan/README.md) (CVPR'2020)
- ✅ [BasicVSR](configs/basicvsr/README.md) (CVPR'2021)
- ✅ [IconVSR](configs/iconvsr/README.md) (CVPR'2021)
- ✅ [BasicVSR++](configs/basicvsr_pp/README.md) (CVPR'2022)
- ✅ [RealBasicVSR](configs/real_basicvsr/README.md) (CVPR'2022)

</details>

<details open>
<summary>视频插帧</summary>

- ✅ [TOFlow](configs/tof/README.md) (IJCV'2019)
- ✅ [CAIN](configs/cain/README.md) (AAAI'2020)
- ✅ [FLAVR](configs/flavr/README.md) (CVPR'2021)

</details>

<details open>
<summary>图像上色</summary>

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

- ✅ [Disco-Diffusion](configs/disco_diffusion/README.md)

</details>

<details open>

<summary>3D-aware Generation</summary>

- ✅ [EG3D](configs/eg3d/README.md)

</details>

请参考[模型库](https://mmediting.readthedocs.io/zh_CN/1.x/3_model_zoo.html)了解详情。

## 参与贡献

感谢您为改善 MMEditing 所做的所有贡献。请参阅 MMCV 中的 [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/tree/2.x/CONTRIBUTING.md) 和 MMEngine 中的 [CONTRIBUTING.md](https://github.com/open-mmlab/mmengine/blob/main/CONTRIBUTING_zh-CN.md) 以获取贡献指南。

## 致谢

MMEditing 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

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

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab MMEngine.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具箱
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
