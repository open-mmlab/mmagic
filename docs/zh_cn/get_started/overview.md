# 概述（待更新）
＃ 概述

欢迎来到 MMEditing！ 在本节中，您将了解

- [MMEditing是什么？](#mmediting-是什么)
- [为什么要使用 MMEditing？](#为什么要使用-mmediting)
- [新手入门](#新手入门)
- [基础教程](#基础教程)
- [进阶教程](#进阶教程)

## MMEditing 是什么？

MMEditing 是一个供专业人工智能研究人员和机器学习工程师去处理、编辑和合成图像与视频的开源工具箱。

MMEditing 允许研究人员和工程师使用最先进的预训练模型，并且可以轻松训练和开发新的定制模型。

MMEditing 支持各种基础生成模型，包括：

- 无条件生成对抗网络 (GANs)
- 条件生成对抗网络 (GANs)
- 内部学习
- 扩散模型
- 还有许多其他生成模型即将推出！

MMEditing 支持各种应用程序，包括：

- 图像超分辨率
- 视频超分辨率
- 视频帧插值
- 图像修复
- 图像抠图
- 图像到图像的翻译
- 还有许多其他应用程序即将推出！

<div align=center>
   <img src="https://user-images.githubusercontent.com/12756472/158984079-c4754015-c1f6-48c5-ac46-62e79448c372.jpg"/>
</div>
</br>

<div align=center>
     <video width="100%" controls>
         <source src="https://user-images.githubusercontent.com/12756472/175944645-cabe8c2b-9f25-440b-91cc-cdac4e752c5a.mp4" type="video/mp4">
         <object data="https://user-images.githubusercontent.com/12756472/175944645-cabe8c2b-9f25-440b-91cc-cdac4e752c5a.mp4" width="100%">
         </object>
     </video>
</div>
</br>

<div  align=center>
<video width="100%" 控件>
     <source src="https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4" type="video/mp4">
     <object src="https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4" width="100%">
     </bject>
</video>
</div>

<div align="center">
   <b>StyleGAN3 Images</b>
   <br/>
   <img src="https://user-images.githubusercontent.com/22982797/150450502-c182834f-796f-4397-bd38-df1efe4a8a47.png" width="800"/>
</div>

<div align="center">
   <b>BigGAN Images</b>
   <br/>
   <img src="https://user-images.githubusercontent.com/22982797/127615534-6278ce1b-5cff-4189-83c6-9ecc8de08dfc.png" width="800"/>
</div>

## 为什么要使用 MMEditing？

- **最先进的**

   MMEditing 提供最先进的生成模型来处理、编辑和合成图像和视频。

- **强大而流行的应用程序**

   MMEditing 支持流行和现代的*修复*、*抠图*、*超分辨率* 和*生成* 应用程序。 具体来说，MMEditing 支持 GAN插值、GAN投影、GAN操作和许多其他流行的GAN应用程序。 是时候玩转你的GAN了！

- **全新模块化设计，灵活组合：**

   我们将编辑框架分解为不同的模块，通过组合不同的模块可以轻松构建定制的编辑框架。 具体来说，提出了一种新的复杂损失模块设计，用于自定义模块之间的链接，可以实现不同模块之间的灵活组合。([损失函数](../howto/losses.md))

- **高效的分布式训练：**

   在[MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py)的支持下，可以轻松实现动态架构的分布式训练。

## 开始

安装说明见[安装](install.md)。

＃＃ 用户指南

对于初学者，我们建议从 [基础教程](../user_guides/config.md) 学习 MMEditing 的基本用法。

### 高级指南

对于熟悉MMEditing的用户，可能想了解MMEditing的设计，以及如何扩展repo，如何使用多个repos等高级用法，请参考[高级指导](../advanced_guides/evaluator.md)。

## 开发指南

想要使用 MMEditing 进行深度开发的用户，可以参考[开发指南](../howto/models.md)。