# 变更日志

## v1.0.1 (26/05/2023)

**新功能和改进**

- 支持 StableDiffusion tomesd 加速. [#1801](https://github.com/open-mmlab/mmagic/pull/1801)
- 支持所有 inpainting/matting/image restoration 模型的 inferencer. [#1833](https://github.com/open-mmlab/mmagic/pull/1833), [#1873](https://github.com/open-mmlab/mmagic/pull/1873)
- 支持 animated drawings. [#1837](https://github.com/open-mmlab/mmagic/pull/1837)
- 支持 Style-Based Global Appearance Flow for Virtual Try-On at projects. [#1786](https://github.com/open-mmlab/mmagic/pull/1786)
- 支持 tokenizer wrapper 和 EmbeddingLayerWithFixe. [#1846](https://github.com/open-mmlab/mmagic/pull/1846)

**Bug 修复**

- 修复安装依赖. [#1819](https://github.com/open-mmlab/mmagic/pull/1819)
- 修复 inst-colorization PackInputs. [#1828](https://github.com/open-mmlab/mmagic/pull/1828), [#1827](https://github.com/open-mmlab/mmagic/pull/1827)
- 修复 pip install 时 inferencer 无法使用的问题. [#1875](https://github.com/open-mmlab/mmagic/pull/1875)

## v1.0.0 (25/04/2023)

我们正式发布 MMagic v1.0.0 版本，源自 [MMEditing](https://github.com/open-mmlab/mmediting) 和 [MMGeneration](https://github.com/open-mmlab/mmgeneration)。

![mmagic-log](https://user-images.githubusercontent.com/49083766/233557648-9034f5a0-c85d-4092-b700-3a28072251b6.png)

自从 MMEditing 诞生以来，它一直是许多图像超分辨率、编辑和生成任务的首选算法库，帮助多个研究团队取得 10 余 项国际顶级赛事的胜利，支撑了 100 多个 GitHub 生态项目。经过 OpenMMLab 2.0 框架的迭代更新以及与 MMGeneration 的合并，MMEditing 已经成为了一个支持基于 GAN 和 CNN 的底层视觉算法的强大工具。

而今天，MMEditing 将更加拥抱生成式 AI（Generative AI），正式更名为 **MMagic**（**M**ultimodal **A**dvanced, **G**enerative, and **I**ntelligent **C**reation），致力于打造更先进、更全面的 AIGC 开源算法库。

在 MMagic 中，我们已经支持了 53+ 模型，分布于 Stable Diffusion 的微调、图文生成、图像及视频修复、超分辨率、编辑和生成等多种任务。配合 [MMEngine](https://github.com/open-mmlab/mmengine) 出色的训练与实验管理支持，MMagic 将为广大研究者与 AIGC 爱好者们提供更加快捷灵活的实验支持，助力你的 AIGC 探索之旅。使用 MMagic，体验更多生成的魔力！让我们一起开启超越编辑的新纪元！ More than Editing, Unlock the Magic!

**主要更新**

**1. 新算法**

我们支持了4个新任务以及11个新算法。

- Text2Image / Diffusion
  - ControlNet
  - DreamBooth
  - Stable Diffusion
  - Disco Diffusion
  - GLIDE
  - Guided Diffusion
- 3D-aware Generation
  - EG3D
- Image Restoration
  - NAFNet
  - Restormer
  - SwinIR
- Image Colorization
  - InstColorization

https://user-images.githubusercontent.com/49083766/233564593-7d3d48ed-e843-4432-b610-35e3d257765c.mp4

**2. Magic Diffusion Model**

针对 Diffusion Model，我们提供了以下“魔法”

- 支持基于 Stable Diffusion 与 Disco Diffusion 的图像生成.

- 支持 Dreambooth 以及 DreamBooth LoRA 等 Finetune 方法.

- 支持 ControlNet 进行可控性的文本到图像生成.
  ![de87f16f-bf6d-4a61-8406-5ecdbb9167b6](https://user-images.githubusercontent.com/49083766/233558077-2005e603-c5a8-49af-930f-e7a465ca818b.png)

- 支持 xFormers 加速和优化策略，提高训练与推理效率.

- 支持基于 MultiFrame Render 的视频生成.
  MMagic 支持通过 ControlNet 与多帧渲染法实现长视频的生成。
  prompt keywords: a handsome man, silver hair, smiling, play basketball

  https://user-images.githubusercontent.com/12782558/227149757-fd054d32-554f-45d5-9f09-319184866d85.mp4

  prompt keywords: a girl, black hair, white pants, smiling, play basketball

  https://user-images.githubusercontent.com/49083766/233559964-bd5127bd-52f6-44b6-a089-9d7adfbc2430.mp4

  prompt keywords: a handsome man

  https://user-images.githubusercontent.com/12782558/227152129-d70d5f76-a6fc-4d23-97d1-a94abd08f95a.mp4

- 支持通过 Wrapper 调用 Diffusers 的基础模型以及采样策略.

- SAM + MMagic = Generate Anything！
  当下流行的 SAM（Segment Anything Model）也可以为 MMagic 提供更多加持！想制作自己的动画，可以移步至 [OpenMMLab PlayGround](https://github.com/open-mmlab/playground/blob/main/mmediting_sam/README.md)！

  https://user-images.githubusercontent.com/49083766/233562228-f39fc675-326c-4ae8-986a-c942059effd0.mp4

**3. 框架升级**

为了提升你的“施法”效率，我们对“魔术回路”做了以下升级:

- 通过 OpenMMLab 2.0 框架的 MMEngine 和 MMCV， MMagic 将编辑框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。我们可以像搭建“乐高”一样定义训练流程，提供丰富的组件和策略。在 MMagic 中，你可以使用不同的 APIs 完全控制训练流程.
- 支持 33+ 算法 Pytorch 2.0 加速.
- 重构 DataSample，支持 batch 维度的组合与拆分.
- 重构 DataPreprocessor，并统一各种任务在训练与推理时的数据格式.
- 重构 MultiValLoop 与 MultiTestLoop，同时支持生成类型指标（e.g. FID）与重建类型指标（e.g. SSIM） 的评测，同时支持一次性评测多个数据集
- 支持本地可视化以及使用 tensorboard 或 wandb的可视化.

**新功能和改进**

- 支持 53+ 算法，232+ 配置，213+ 模型权重，26+ 损失函数，and 20+ 评价指标.
- 支持 controlnet 动画生成以及 Gradio gui. [点击查看.](https://github.com/open-mmlab/mmagic/tree/main/configs/controlnet_animation)
- 支持 Inferencer 和 Demo，使用High-level Inference APIs. [点击查看.](https://github.com/open-mmlab/mmagic/tree/main/demo)
- 支持 Inpainting 推理的 Gradio gui. [点击查看.](https://github.com/open-mmlab/mmagic/blob/main/demo/gradio-demo.py)
- 支持可视化图像/视频质量比较工具. [点击查看.](https://github.com/open-mmlab/mmagic/tree/main/tools/gui)
- 开启 projects，助力社区更快向算法库中添加新算法. [点击查看.](https://github.com/open-mmlab/mmagic/tree/main/projects)
- 完善数据集的预处理脚本和使用说明文档. [点击查看.](https://github.com/open-mmlab/mmagic/tree/main/tools/dataset_converters)

## v1.0.0rc7 (07/04/2023)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc7 版本。 此版本支持了 MMEditing 和 MMGeneration 的 51+ 模型，226+ configs 和 212+ checkpoints。以下是此次版本发布的重点新功能

- 支持了 DiffuserWrapper.
- 支持了 ControlNet 的推理与训练.
- 支持了 PyTorch 2.0.

**新功能和改进**

- 支持了 DiffuserWrapper. [#1692](https://github.com/open-mmlab/mmagic/pull/1692)
- 支持了 ControlNet 的推理与训练. [#1744](https://github.com/open-mmlab/mmagic/pull/1744)
- 支持了 PyTorch 2.0 (使用 'inductor' 后端成功编译 33+ 模型) [#1742](https://github.com/open-mmlab/mmagic/pull/1742).
- 支持了图像超分和视频超分的 inferencer. [#1662](https://github.com/open-mmlab/mmagic/pull/1662), [#1720](https://github.com/open-mmlab/mmagic/pull/1720)
- 重构 get_flops 脚本. [#1675](https://github.com/open-mmlab/mmagic/pull/1675)
- 重构数据集的 dataset_converters 脚本和使用文档. [#1690](https://github.com/open-mmlab/mmagic/pull/1690)
- 迁移 stylegan 算子到 MMCV 中. [#1383](https://github.com/open-mmlab/mmagic/pull/1383)

**Bug 修复**

- 修复 disco inferencer. [#1673](https://github.com/open-mmlab/mmagic/pull/1673)
- 修复 nafnet optimizer 配置. [#1716](https://github.com/open-mmlab/mmagic/pull/1716)
- 修复 tof typo. [#1711](https://github.com/open-mmlab/mmagic/pull/1711)

**贡献者**

@LeoXing1996, @Z-Fran, @plyfager, @zengyh1900, @liuwenran, @ryanxingql, @HAOCHENYE, @VongolaWu

## v1.0.0rc6 (02/03/2023)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc6 版本。 此版本支持了 MMEditing 和 MMGeneration 的 50+ 模型，222+ configs 和 209+ checkpoints。以下是此次版本发布的重点新功能

- 支持了 Inpainting 任务推理的 Gradio gui.
- 支持了图像上色、图像翻译和 GAN 模型的 inferencer.

**新功能和改进**

- 重构了 FileIO. [#1572](https://github.com/open-mmlab/mmagic/pull/1572)
- 重构了 registry. [#1621](https://github.com/open-mmlab/mmagic/pull/1621)
- 重构了 Random degradations. [#1583](https://github.com/open-mmlab/mmagic/pull/1583)
- 重构了 DataSample, DataPreprocessor, Metric 和 Loop. [#1656](https://github.com/open-mmlab/mmagic/pull/1656)
- 使用 mmengine.basemodule 替换 nn.module. [#1491](https://github.com/open-mmlab/mmagic/pull/1491)
- 重构了算法库主页. [#1609](https://github.com/open-mmlab/mmagic/pull/1609)
- 支持了 Inpainting 任务推理的 Gradio gui. [#1601](https://github.com/open-mmlab/mmagic/pull/1601)
- 支持了图像上色的 inferencer. [#1588](https://github.com/open-mmlab/mmagic/pull/1588)
- 支持了图像翻译和所有 GAN 模型的 inferencer. [#1650](https://github.com/open-mmlab/mmagic/pull/1650)
- 支持了 GAN 模型的 inferencer. [#1653](https://github.com/open-mmlab/mmagic/pull/1653), [#1659](https://github.com/open-mmlab/mmagic/pull/1659)
- 新增 Print config 工具. [#1590](https://github.com/open-mmlab/mmagic/pull/1590)
- 改进 type hints. [#1604](https://github.com/open-mmlab/mmagic/pull/1604)
- 更新 metrics 和 datasets 的中文文档. [#1568](https://github.com/open-mmlab/mmagic/pull/1568), [#1638](https://github.com/open-mmlab/mmagic/pull/1638)
- 更新 BigGAN 和 Disco-Diffusion 的中文文档. [#1620](https://github.com/open-mmlab/mmagic/pull/1620)
- 更新 Guided-Diffusion 的 Evaluation 和 README. [#1547](https://github.com/open-mmlab/mmagic/pull/1547)

**Bug 修复**

- 修复 EMA `momentum`. [#1581](https://github.com/open-mmlab/mmagic/pull/1581)
- 修复 RandomNoise 的输出类型. [#1585](https://github.com/open-mmlab/mmagic/pull/1585)
- 修复 pytorch2onnx 工具. [#1629](https://github.com/open-mmlab/mmagic/pull/1629)
- 修复 API 文档. [#1641](https://github.com/open-mmlab/mmagic/pull/1641), [#1642](https://github.com/open-mmlab/mmagic/pull/1642)
- 修复 RealESRGAN 加载 EMA 参数. [#1647](https://github.com/open-mmlab/mmagic/pull/1647)
- 修复 dataset_converters 脚本的 arg passing bug. [#1648](https://github.com/open-mmlab/mmagic/pull/1648)

**贡献者**

@plyfager, @LeoXing1996, @Z-Fran, @zengyh1900, @VongolaWu, @liuwenran, @austinmw, @dienachtderwelt, @liangzelong, @i-aki-y, @xiaomile, @Li-Qingyun, @vansin, @Luo-Yihang, @ydengbi, @ruoningYu, @triple-Mu

## v1.0.0rc5 (04/01/2023)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc5 版本。 此版本支持了 MMEditing 和 MMGeneration 的 49+ 模型，180+ configs 和 177+ checkpoints。以下是此次版本发布的重点新功能

- 支持了 Restormer 算法.
- 支持了 GLIDE 算法.
- 支持了 SwinIR 算法.
- 支持了 Stable Diffusion 算法.

**新功能和改进**

- 新增 Disco notebook. (#1507)
- 优化测试 requirements 和 CI. (#1514)
- 自动生成文档 summary 和 API docstring. (#1517)
- 开启 projects. (#1526)
- 支持 mscoco dataset. (#1520)
- 改进中文文档. (#1532)
- 添加 Type hints. (#1481)
- 更新模型权重下载链接. (#1554)
- 更新部署指南. (#1551)

**Bug 修复**

- 修复文档链接检查. (#1522)
- 修复 ssim bug. (#1515)
- 修复 realesrgan 的 `extract_gt_data`. (#1542)
- 修复算法索引. (#1559)
- F修复 disco-diffusion 的 config 路径. (#1553)
- Fix text2image inferencer. (#1523)

**贡献者**

@plyfager, @LeoXing1996, @Z-Fran, @zengyh1900, @VongolaWu, @liuwenran, @AlexZou14, @lvhan028, @xiaomile, @ldr426, @austin273, @whu-lee, @willaty, @curiosity654, @Zdafeng, @Taited

## v1.0.0rc4 (05/12/2022)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc4 版本。 此版本支持了 MMEditing 和 MMGeneration 的 45+ 模型，176+ configs 和 175+ checkpoints。以下是此次版本发布的重点新功能

- 支持了 High-level APIs.
- 支持了 diffusion 算法.
- 支持了 Text2Image 任务.
- 支持了 3D-Aware Generation.

**新功能和改进**

- 支持和重构了 High-level APIs. (#1410)
- 支持了 disco-diffusion 文生图算法. (#1234, #1504)
- 支持了 EG3D 算法. (#1482, #1493, #1494, #1499)
- 支持了 NAFNet 算法. (#1369)

**Bug 修复**

- 修复 srgan 的训练配置. (#1441)
- 修复 cain 的 config. (#1404)
- 修复 rdn 和 srcnn 的训练配置. (#1392)

**贡献者**

@plyfager, @LeoXing1996, @Z-Fran, @zengyh1900, @VongolaWu, @gaoyang07, @ChangjianZhao, @zxczrx123, @jackghosts, @liuwenran, @CCODING04, @RoseZhao929, @shaocongliu, @liangzelong.

## v1.0.0rc3 (10/11/2022)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc3 版本。 此版本支持了 MMEditing 和 MMGeneration 的 43+ 模型，170+ configs 和 169+ checkpoints。以下是此次版本发布的重点新功能

- 将 `mmdet` 和 `clip` 改为可选安装需求.

**新功能和改进**

- 支持 `mmdet` 的 `try_import`. (#1408)
- 支持 `flip` 的 `try_import`. (#1420)
- 更新 `.gitignore`. ($1416)
- 设置 `inception_utils` 的 `real_feat` 为 cpu 变量. (#1415)
- 更新 StyleGAN2 和 PEGAN 的 README 和 configs.  (#1418)
- 改进 API 文档的渲染. (#1373)

**Bug 修复**

- 修复 ESRGAN 的 config 和预训练模型加载. (#1407)
- 修复 LSGAN 的 config. (#1409)
- 修复 CAIN 的 config. (#1404)

**贡献者**

@Z-Fran, @zengyh1900, @plyfager, @LeoXing1996, @ruoningYu.

## v1.0.0rc2 (02/11/2022)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc2 版本。 此版本支持了 MMEditing 和 MMGeneration 的 43+ 模型，170+ configs 和 169+ checkpoints。以下是此次版本发布的重点新功能

- 基于 patch 和 slider 的 图像和视频可视化质量比较工具.
- 支持了图像上色算法.

**新功能和改进**

- 支持了质量比较工具. (#1303)
- 支持了 instance aware colorization 上色算法. (#1370)
- 支持使用不同采样模型的 multi-metrics. (#1171)
- 改进代码实现
  - 重构 evaluation metrics. (#1164)
  - 在 PGGAN 的 `forward` 中保存 gt 图像. (#1332)
  - 改进 `preprocess_div2k_dataset.py` 脚本的默认参数. (#1380)
  - 支持在 visualizer 中裁剪像素值. (#1365)
  - 支持了 SinGAN 数据集和 SinGAN demo. (#1363)
  - 支持 GenDataPreprocessor 中返回 int 和 float 数据类型. (#1385)
- 改进文档
  - 更新菜单切换. (#1162)
  - 修复 TTSR README. (#1325)

**Bug 修复**

- 修复 PPL bug. (#1172)
- 修复 RDN `number of channels` 参数. (#1328)
- 修复 demo 的 exceptions 类型. (#1372)
- 修复 realesrgan ema. (#1341)
- 改进 assertion 检查 `GenerateFacialHeatmap` 的数据类型为 `np.float32`. (#1310)
- 修复 `unpaired_dataset.py` 的采样方式. (#1308)
- 修复视频超分模型的 pytorch2onnx 脚本. (#1300)
- 修复错误的 config 配置. (#1167,#1200,#1236,#1293,#1302,#1304,#1319,#1331,#1336,#1349,#1352,#1353,#1358,#1364,#1367,#1384,#1386,#1391,#1392,#1393)

**贡献者**

@LeoXing1996, @Z-Fran, @zengyh1900, @plyfager, @ryanxingql, @ruoningYu, @gaoyang07.

## v1.0.0rc1(23/9/2022)

MMEditing 1.0.0rc1 已经合并了 MMGeneration 1.x。

- 支持 42+ 算法, 169+ 配置文件 and 168+ 预训练模型参数文件.
- 支持 26+ loss functions, 20+ metrics.
- 支持 tensorboard, wandb.
- 支持 unconditional GANs, conditional GANs, image2image translation 以及 internal learning.

## v1.0.0rc0(31/8/2022)

MMEditing 1.0.0rc0 是 MMEditing 1.x 的第一个版本，是 OpenMMLab 2.0 项目的一部分。

基于新的[训练引擎](https://github.com/open-mmlab/mmengine), MMEditing 1.x 统一了数据、模型、评测和可视化的接口。

该版本存在有一些 BC-breaking 的修改。 请在[迁移指南](https://mmagic.readthedocs.io/zh_CN/latest/migration/overview.html)中查看更多细节。
