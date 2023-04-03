<div id="top" align="center">
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
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmediting.readthedocs.io/zh_CN/main/)
[![badge](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![license](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)

[📘使用文档](https://mmediting.readthedocs.io/en/main/) |
[🛠️安装教程](https://mmediting.readthedocs.io/zh_CN/main/get_started/install.html) |
[📊模型库](https://mmediting.readthedocs.io/zh_CN/main/model_zoo/overview.html) |
[🆕更新记录](https://mmediting.readthedocs.io/zh_CN/main/changelog.html) |
[🚀进行中的项目](https://github.com/open-mmlab/mmediting/projects) |
[🤔提出问题](https://github.com/open-mmlab/mmediting/issues)

[English](README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

## 🚀 最新进展 <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

### 最新的 [**MMEditing v1.0.0rc6**](https://github.com/open-mmlab/mmediting/releases/tag/v1.0.0rc6) 版本已经在 \[02/03/2023\] 发布:

- 支持了 Inpainting 任务推理的 Gradio gui.
- 支持了图像上色、图像翻译和 GAN 模型的 inferencer.

**MMEditing** 已经支持了[MMGeneration](https://github.com/open-mmlab/mmgeneration)中的全量任务、模型、优化函数和评价指标 ，并基于[MMEngine](https://github.com/open-mmlab/mmengine)统一了各组件接口 😍。

如果想了解更多版本更新细节和历史信息，请阅读[更新日志](docs/zh_cn/changelog.md)。如果想从[旧版本](https://github.com/open-mmlab/mmediting/tree/master) MMEditing 0.x 迁移到新版本 MMEditing main，请阅读[迁移文档](docs/zh_cn/migration/overview.md)。

## 📄 目录

- [📖 介绍](#📖-介绍)
- [🙌 参与贡献](#🙌-参与贡献)
- [🛠️ 安装](#🛠️-安装)
- [📊 模型库](#📊-模型库)
- [🤝 致谢](#🤝-致谢)
- [🖊️ 引用](#🖊️-引用)
- [🎫 许可证](#🎫-许可证)
- [🏗️ ️OpenMMLab 的其他项目](#🏗️-️openmmlab-的其他项目)

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📖 介绍

MMEditing 是基于 PyTorch 的图像&视频编辑和生成开源工具箱。是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一。

目前 MMEditing 支持多种图像和视频的生成/编辑任务。

https://user-images.githubusercontent.com/12782558/217152698-49169038-9872-4200-80f7-1d5f7613afd7.mp4

主分支代码的最佳实践基于 **Python 3.8+** 和 **PyTorch 1.9+** 。

### ✨ 主要特性

- **SOTA**

  MMEditing 提供了处理、编辑、生成图像和视频的SOTA算法。

- **强有力且流行的应用**

  MMEditing 支持了流行的图像修复、图文生成、3D生成、图像修补、抠图、超分辨率和生成等任务的应用。特别是 MMEditing 支持了 GANs 的插值，投影和编辑和其他流行的应用。请用你的 GANs 尽情尝试！

- **灵活组合的模块化设计**

  MMEditing 将编辑框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。

- **高效的分布式训练**

  得益于 [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), 动态模型结构的分布式训练可以轻松实现。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🙌 参与贡献

越来越多社区贡献者的加入使我们的算法库日益发展。最近由社区贡献的项目包括：

- [GLIDE](projects/glide/configs/README.md) 来自 @Taited.
- [Restormer](configs/restormer/README.md) 来自 @AlexZou14.
- [SwinIR](configs/swinir/README.md) 来自 @Zdafeng.

为使向 MMEditing 中添加项目更加容易，我们开启了 [Projects](projects/README.md) 。

感谢您为改善 MMEditing 所做的所有贡献。请参阅 MMCV 中的 [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/tree/2.x/CONTRIBUTING.md) 和 MMEngine 中的 [CONTRIBUTING.md](https://github.com/open-mmlab/mmengine/blob/main/CONTRIBUTING_zh-CN.md) 以获取贡献指南。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🛠️ 安装

MMEditing 依赖 [PyTorch](https://pytorch.org/)，[MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)，以下是安装的简要步骤。

**步骤 1.**
依照[官方教程](https://pytorch.org/get-started/locally/)安装 PyTorch 。

**步骤 2.**
使用 [MIM](https://github.com/open-mmlab/mim) 安装 MMCV 。

```
pip3 install openmim
# wait for more pre-compiled pkgs to release
mim install 'mmcv>=2.0.0rc1'
```

**步骤 3.**
从源码安装 MMEditing

```
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

更详细的安装指南请参考 [安装指南](docs/zh_cn/get_started/install.md) 。

**开始使用**

请参考[快速运行](docs/zh_cn/get_started/quick_run.md)和[推理演示](docs/zh_cn/user_guides/inference.md)获取MMEditing的基本用法。

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📊 模型库

<div align="center">
  <b>支持的算法</b>
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
            <li><a href="configs/sngan_proj/README.md">SNGAN/Projection GAN (ICLR'2018)</a></li>
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
          <li><a href="configs/inst_colorization/README.md">InstColorization (CVPR'2020)</a></li>
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

请参考[模型库](https://mmediting.readthedocs.io/zh_CN/main/model_zoo/overview.html)了解详情。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🤝 致谢

MMEditing 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

<a href="https://github.com/open-mmlab/mmediting/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liuwenran/mmediting" />
</a>

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🖊️ 引用

如果 MMEditing 对您的研究有所帮助，请按照如下 bibtex 引用它。

```bibtex
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🎫 许可证

本项目开源自 [Apache 2.0 license](LICENSE)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🏗️ ️OpenMMLab 的其他项目

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

<p align="right"><a href="#top">🔝返回顶部</a></p>

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
