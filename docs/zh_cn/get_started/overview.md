# 概述

欢迎来到 MMagic！ 在本节中，您将了解

- [MMagic是什么？](#mmagic-是什么)
- [为什么要使用 MMagic？](#为什么要使用-mmagic)
- [新手入门](#新手入门)
- [基础教程](#基础教程)
- [进阶教程](#进阶教程)

## MMagic 是什么？

MMagic (**M**ultimodal **A**dvanced, **G**enerative, and **I**ntelligent **C**reation) 是一个供专业人工智能研究人员和机器学习工程师去处理、编辑和生成图像与视频的开源 AIGC 工具箱。

MMagic 允许研究人员和工程师使用最先进的预训练模型，并且可以轻松训练和开发新的定制模型。

MMagic 支持各种基础生成模型，包括：

- 无条件生成对抗网络 (GANs)
- 条件生成对抗网络 (GANs)
- 内部学习
- 扩散模型
- 还有许多其他生成模型即将推出！

MMagic 支持各种应用程序，包括：

- 图文生成
- 图像翻译
- 3D 生成
- 图像超分辨率
- 视频超分辨率
- 视频插帧
- 图像补全
- 图像抠图
- 图像修复
- 图像上色
- 图像生成
- 还有许多其他应用程序即将推出！

<div align=center>
    <video width="100%" controls>
        <source src="https://user-images.githubusercontent.com/49083766/233564593-7d3d48ed-e843-4432-b610-35e3d257765c.mp4" type="video/mp4">
        <object data="https://user-images.githubusercontent.com/49083766/233564593-7d3d48ed-e843-4432-b610-35e3d257765c.mp4" width="100%">
        </object>
    </video>
</div>
</br>

## 为什么要使用 MMagic？

- **SOTA 算法**

  MMagic 提供了处理、编辑、生成图像和视频的 SOTA 算法。

- **强有力且流行的应用**

  MMagic 支持了流行的图像修复、图文生成、3D生成、图像修补、抠图、超分辨率和生成等任务的应用。特别是 MMagic 支持了 Stable Diffusion 的微调和许多激动人心的 diffusion 应用，例如 ControlNet 动画生成。MMagic 也支持了 GANs 的插值，投影，编辑和其他流行的应用。请立即开始你的 AIGC 探索之旅！

- **高效的框架**

  通过 OpenMMLab 2.0 框架的 MMEngine 和 MMCV， MMagic 将编辑框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。我们可以像搭建“乐高”一样定义训练流程，提供丰富的组件和策略。在 MMagic 中，你可以使用不同的 APIs 完全控制训练流程。得益于 [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), 动态模型结构的分布式训练可以轻松实现。

## 新手入门

安装说明见[安装](install.md)。

## 基础教程

对于初学者，我们建议从 [基础教程](../user_guides/config.md) 学习 MMagic 的基本用法。

## 进阶教程

对于熟悉 MMagic 的用户，可能想了解 MMagic 的进阶实用，以及如何扩展算法库，如何使用多个算法库框架等高级用法，请参考[进阶教程](../advanced_guides/evaluator.md)。

## 开发指南

想要使用 MMagic 进行深度开发的用户，可以参考[开发指南](../howto/models.md)。
