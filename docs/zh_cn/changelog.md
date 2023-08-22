# 变更日志

<<<<<<< HEAD
## v1.0.1 (26/05/2023)
=======
## v0.16.1 (24/02/2023)

**新特性和改进**

- 支持新评价指标 FID 和 KID。 [#775](https://github.com/open-mmlab/mmediting/pull/775)
- 支持 ResidualBlockNoBN 模块设置 `groups` 参数。 [#1510](https://github.com/open-mmlab/mmediting/pull/1510)

**Bug 修复**

- 修复 TTSR 配置文件的 Bug。 [#1435](https://github.com/open-mmlab/mmediting/pull/1435)
- 修复 RealESRGAN 测试数据集配置。 [#1489](https://github.com/open-mmlab/mmediting/pull/1489)
- 修复训练脚本储存 config 文件的功能。 [#1584](https://github.com/open-mmlab/mmediting/pull/1584)
- 修复 `pixel-unshuffle` 模块动态输入导出 ONNX 的Bug。 [#1637](https://github.com/open-mmlab/mmediting/pull/1637)

**贡献者**

一共有 10 位开发者对本次发布做出贡献。感谢 @LeoXing1996, @Z-Fran, @zengyh1900, @liuky74, @KKIEEK, @zeakey, @Sqhttwl, @yhna940, @gihwan-kim, @vansin。

## v0.16.0 (31/10/2022)

**接口变更**

`VisualizationHook` 即将废弃，建议用户使用 `MMEditVisualizationHook`。(#1375)

<table align="center">
  <thead>
      <tr align='center'>
          <td>旧版本</td>
          <td>新版本</td>
      </tr>
  </thead>
  <tbody><tr valign='top'>
  <th>

```python
visual_config = dict(  # 构建可视化钩子的配置
  type='VisualizationHook',
  output_dir='visual',
  interval=1000,
  res_name_list=[
      'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
  ],
)
```

</th>
  <th>

```python
visual_config = dict(  # 构建可视化钩子的配置
  type='MMEditVisualizationHook',
  output_dir='visual',
  interval=1000,
  res_name_list=[
      'gt_img', 'masked_img', 'fake_res', 'fake_img', 'fake_gt_local'
  ],
)
```

</th></tr>
</tbody></table>

**改进**

- 改进 `preprocess_div2k_dataset.py` 中的参数类型。 (#1381)
- 更新 RDN 的 docstring。 (#1326)
- 更新 `readme.md` 中的介绍说明。 (#)

**Bug 修复**

- 修复 FLAVR 在 `mmedit/models/video_interpolators` 中的注册和使用。(#1186)
- 修复 `restoration_video_inference.py` 中的路径处理问题。 (#1262)
- 修正 RDB 模型结构中的卷积通道数。(#1292, #1311)

**Contributors**

一共有 5 位 开发者对本次发布做出贡献。感谢 @LeoXing1996, @Z-Fran, @zengyh1900, @ryanxingql, @ruoningYu。

## v0.15.2 (09/09/2022)

**改进**

- \[Docs\] 修正文档中的拼写错误 by @Yulv-git in https://github.com/open-mmlab/mmediting/pull/1079
- \[Docs\] 修正 model zoo 数据集的链接 by @Z-Fran in https://github.com/open-mmlab/mmediting/pull/1043
- \[Docs\] 修正 readme 中的拼写错误 by @arch-user-france1 in https://github.com/open-mmlab/mmediting/pull/1078
- \[Improve\] 提供 FLAVR demo by @Yshuo-Li in https://github.com/open-mmlab/mmediting/pull/954
- \[Fix\] 更新 MMCV 的版本上限到 1.7 by @wangruohui in https://github.com/open-mmlab/mmediting/pull/1001
- \[Improve\] 改进 niqe_pris_params.npz 安装路径 by @ychfan in https://github.com/open-mmlab/mmediting/pull/995
- \[CI\] 更新 Github Actions， CircleCI 以及 Issue 和 PR 的模板 by @zengyh1900 in https://github.com/open-mmlab/mmediting/pull/1087

**Contributors**

@wangruohui @Yshuo-Li @zengyh1900 @Z-Fran @ychfan @arch-user-france1 @Yulv-git

## v0.15.1 (04/07/2022)

**Bug 修复**

- \[修复\] 更新 cain_b5_g1b32_vimeo90k_triplet.py 配置文件 ([#929](https://github.com/open-mmlab/mmediting/pull/929))
- \[文档\] 修复 OST 数据集的链接 ([#933](https://github.com/open-mmlab/mmediting/pull/933))

**改进**

- \[文档\] 更新 OST 数据集指令 ([#937](https://github.com/open-mmlab/mmediting/pull/937))
- \[测试\] 在 CUDA 环境中没有实际执行 ([#921](https://github.com/open-mmlab/mmediting/pull/921))
- \[文档\] 首页演示视频添加水印 ([#935](https://github.com/open-mmlab/mmediting/pull/935))
- \[测试\] 添加 mim ci ([#928](https://github.com/open-mmlab/mmediting/pull/928))
- \[文档\] 更新 FLAVR 的 README.md ([#919](https://github.com/open-mmlab/mmediting/pull/919))
- \[改进\] 更新 .pre-commit-config.yaml 中的 md-format ([#917](https://github.com/open-mmlab/mmediting/pull/917))
- \[改进\] 在 setup.py 中添加 miminstall.txt ([#916](https://github.com/open-mmlab/mmediting/pull/916))
- \[修复\] 修复 dim/README.md 中的混乱问题 ([#913](https://github.com/open-mmlab/mmediting/pull/913))
- \[改进\] 跳过有问题的 opencv-python 版本 ([#833](https://github.com/open-mmlab/mmediting/pull/833))

**贡献者**

@wangruohui @Yshuo-Li

## v0.15.0 (01/06/2022)

**Highlights主要更新**

1. 支持 FLAVR
2. 支持 AOT-GAN
3. 在 CAIN 中支持 ReduceLROnPlateau 策略

**新功能**

- 支持 AOT-GAN ([#681](https://github.com/open-mmlab/mmediting/pull/681))
- 支持 Vimeo90k-triplet 数据集 ([#810](https://github.com/open-mmlab/mmediting/pull/810))
- 为 mm-assistant 添加默认 config ([#827](https://github.com/open-mmlab/mmediting/pull/827))
- 支持 CPU demo ([#848](https://github.com/open-mmlab/mmediting/pull/848))
- 在 `LoadImageFromFileList` 中支持 `use_cache` 和 `backend` ([#857](https://github.com/open-mmlab/mmediting/pull/857))
- 支持 VFIVimeo90K7FramesDataset ([#858](https://github.com/open-mmlab/mmediting/pull/858))
- 在 VFI pipeline 中支持 ColorJitter ([#859](https://github.com/open-mmlab/mmediting/pull/859))
- 支持 ReduceLrUpdaterHook ([#860](https://github.com/open-mmlab/mmediting/pull/860))
- 在 IterBaseRunner 中支持 `after_val_epoch` ([#861](https://github.com/open-mmlab/mmediting/pull/861))
- 支持 FLAVR Net ([#866](https://github.com/open-mmlab/mmediting/pull/866), [#867](https://github.com/open-mmlab/mmediting/pull/867), [#897](https://github.com/open-mmlab/mmediting/pull/897))
- 支持 MAE 评估方式 ([#871](https://github.com/open-mmlab/mmediting/pull/871))
- 使用 mdformat ([#888](https://github.com/open-mmlab/mmediting/pull/888))
- 在 CAIN 中支持 ReduceLROnPlateau 策略 ([#906](https://github.com/open-mmlab/mmediting/pull/906))

**Bug 修复**

- 在 restoration_demo.py 中将 `-` 改为 `_` ([#834](https://github.com/open-mmlab/mmediting/pull/834))
- 移除 requirements/docs.txt 中的 recommonmark ([#844](https://github.com/open-mmlab/mmediting/pull/844))
- 将 README 中的 EDVR 移动到 VSR 类别中 ([#849](https://github.com/open-mmlab/mmediting/pull/849))
- 修改 crop.py，移除跨栏 F-string 中的 `,` ([#855](https://github.com/open-mmlab/mmediting/pull/855))
- 修改 test_pipeline，将重复的 `lq_path` 改为 `gt_path` ([#862](https://github.com/open-mmlab/mmediting/pull/862))
- 修复 TOF-VFI 的 unittest 问题 ([#873](https://github.com/open-mmlab/mmediting/pull/873))
- 解决 VFI demo 中帧序列出错问题 ([#891](https://github.com/open-mmlab/mmediting/pull/891))
- 修复 README 中的 logo & contrib 链接 ([#898](https://github.com/open-mmlab/mmediting/pull/898))
- 修复 indexnet_dimaug_mobv2_1x16_78k_comp1k.py ([#901](https://github.com/open-mmlab/mmediting/pull/901))

**改进**

- 在训练和测试脚本中增加 `--cfg-options` 参数 ([#826](https://github.com/open-mmlab/mmediting/pull/826))
- 更新 MMCV_MAX 到 1.6 ([#829](https://github.com/open-mmlab/mmediting/pull/829))
- 在 README 中更新 TOFlow ([#835](https://github.com/open-mmlab/mmediting/pull/835))
- 恢复 beirf 安装步骤，合并可选要求 ([#836](https://github.com/open-mmlab/mmediting/pull/836))
- 在 citation 中使用 {MMEditing Contributors} ([#838](https://github.com/open-mmlab/mmediting/pull/838))
- 增加定制损失函数的教程 ([#839](https://github.com/open-mmlab/mmediting/pull/839))
- 在 README 中添加安装指南 (wiki ver) ([#845](https://github.com/open-mmlab/mmediting/pull/845))
- 在中文文档中添加“需要帮助翻译”的说明 ([#850](https://github.com/open-mmlab/mmediting/pull/850))
- 在 README_zh-CN.md 中添加微信二维码 ([#851](https://github.com/open-mmlab/mmediting/pull/851))
- 支持 SRFolderVideoDataset 的非零帧索引，修复拼写错误 ([#853](https://github.com/open-mmlab/mmediting/pull/853))
- 创建 docker 的 README.md ([#856](https://github.com/open-mmlab/mmediting/pull/856))
- 优化 IO 流量偏差 ([#881](https://github.com/open-mmlab/mmediting/pull/881))
- 将 wiki/installation 移到 docs ([#883](https://github.com/open-mmlab/mmediting/pull/883))
- 添加 `myst_heading_anchors` ([#887](https://github.com/open-mmlab/mmediting/pull/887))
- 在 inpainting demo 中使用预训练模型链接 ([#892](https://github.com/open-mmlab/mmediting/pull/892))

**贡献者**

@wangruohui @quincylin1 @nijkah @jayagami @ckkelvinchan @ryanxingql @NK-CS-ZZL @Yshuo-Li

## v0.14.0 (01/04/2022)
>>>>>>> 6f2f3ae2ad3e365f94bbf19c01a1d1056dad3895

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

<<<<<<< HEAD
## v1.0.0rc4 (05/12/2022)
=======
## v0.11.0 (03/11/2021)

**亮点**

1. 支持使用 GLEAN 处理人脸图像的盲超分辨率
2. 支持 Real-ESRGAN 模型 #546

**新功能**

- 指数移动平均线钩子 #542
- 支持 DF2K_OST 数据 #566

**改进**

- 增加与 MATLAB 相似的双线性插值算法 #507
- 在训练期间支持随机退化 #504
- 支持 torchserve #568

## v0.10.0 (12/08/2021).

**亮点**

1. 支持 LIIF-RDN (CVPR'2021)
2. 支持 BasicVSR++ (NTIRE'2021)

**新功能**

- Video SR datasets 支持加载文件列表 ([#423](https://github.com/open-mmlab/mmediting/pull/423))
- 支持 persistent worker ([#426](https://github.com/open-mmlab/mmediting/pull/426))
- 支持 LIIF-RDN ([#428](https://github.com/open-mmlab/mmediting/pull/428), [#440](https://github.com/open-mmlab/mmediting/pull/440))
- 支持 BasicVSR++ ([#451](https://github.com/open-mmlab/mmediting/pull/451), [#467](https://github.com/open-mmlab/mmediting/pull/467))
- 支持 mim ([#455](https://github.com/open-mmlab/mmediting/pull/455))

**Bug 修复**

- 修复了 stat.py 中的 bug ([#420](https://github.com/open-mmlab/mmediting/pull/420))
- 修复了 tensor2img 函数中的 astype 错误 ([#429](https://github.com/open-mmlab/mmediting/pull/429))
- 修复了当 pytorch >= 1.7 时由  torch.new_tensor 导致的 device 错误 ([#465](https://github.com/open-mmlab/mmediting/pull/465))
- 修复了 .mmedit/apis/train.py 中的 \_non_dist_train ([#473](https://github.com/open-mmlab/mmediting/pull/473))
- 修复了多节点分布式测试函数 ([#478](https://github.com/open-mmlab/mmediting/pull/478))

**兼容性更新**

- 对 pytorch2onnx 重构了 LIIF  ([#425](https://github.com/open-mmlab/mmediting/pull/425))

**改进**

- 更新了部分中文文档 ([#415](https://github.com/open-mmlab/mmediting/pull/415), [#416](https://github.com/open-mmlab/mmediting/pull/416), [#418](https://github.com/open-mmlab/mmediting/pull/418), [#421](https://github.com/open-mmlab/mmediting/pull/421), [#424](https://github.com/open-mmlab/mmediting/pull/424), [#431](https://github.com/open-mmlab/mmediting/pull/431), [#442](https://github.com/open-mmlab/mmediting/pull/442))
- 添加了 pytorch 1.9.0 的 CI ([#444](https://github.com/open-mmlab/mmediting/pull/444))
- 重写了 README.md 的 configs 文件 ([#452](https://github.com/open-mmlab/mmediting/pull/452))
- 避免在单元测试中加载 VGG 网络的预训练权重 ([#466](https://github.com/open-mmlab/mmediting/pull/466))
- 支持在 div2k 数据集预处理时指定 scales ([#472](https://github.com/open-mmlab/mmediting/pull/472))
- 支持 readthedocs 中的所有格式 ([#479](https://github.com/open-mmlab/mmediting/pull/479))
- 使用 mmcv 的 version_info ([#480](https://github.com/open-mmlab/mmediting/pull/480))
- 删除了 restoration_video_demo.py 中不必要的代码 ([#484](https://github.com/open-mmlab/mmediting/pull/484))
- 将 DistEvalIterHook 的优先级修改为 'LOW' ([#489](https://github.com/open-mmlab/mmediting/pull/489))
- 重置资源限制 ([#491](https://github.com/open-mmlab/mmediting/pull/491))
- 在 README_CN.md 中更新了 QQ 的 QR code  ([#494](https://github.com/open-mmlab/mmediting/pull/494))
- 添加了 `myst_parser` ([#495](https://github.com/open-mmlab/mmediting/pull/495))
- 添加了 license 信息 ([#496](https://github.com/open-mmlab/mmediting/pull/496))
- 修正了 StyleGAN modules 中的拼写错误 ([#427](https://github.com/open-mmlab/mmediting/pull/427))
- 修正了 docs/demo.md 中的拼写错误 ([#453](https://github.com/open-mmlab/mmediting/pull/453), [#454](https://github.com/open-mmlab/mmediting/pull/454))
- 修复了 tools/data/super-resolution/reds/README.md 中的拼写错误 ([#469](https://github.com/open-mmlab/mmediting/pull/469))

## v0.9.0 (30/06/2021).
>>>>>>> 6f2f3ae2ad3e365f94bbf19c01a1d1056dad3895

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

<<<<<<< HEAD
- 修复 srgan 的训练配置. (#1441)
- 修复 cain 的 config. (#1404)
- 修复 rdn 和 srcnn 的训练配置. (#1392)
=======
- 修复了 restoration_video_inference.py 中的 bug ([#379](https://github.com/open-mmlab/mmediting/pull/379))
- 修复了 LIIF 的配置文件 ([#368](https://github.com/open-mmlab/mmediting/pull/368))
- 修改了 pre-trained EDVR-M 的路径 ([#396](https://github.com/open-mmlab/mmediting/pull/396))
- 修复了 restoration_video_inference 中的 normalization ([#406](https://github.com/open-mmlab/mmediting/pull/406))
- 修复了单元测试中的 \[brush_stroke_mask\] 错误 ([#409](https://github.com/open-mmlab/mmediting/pull/409))
>>>>>>> 6f2f3ae2ad3e365f94bbf19c01a1d1056dad3895

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

<<<<<<< HEAD
基于新的[训练引擎](https://github.com/open-mmlab/mmengine), MMEditing 1.x 统一了数据、模型、评测和可视化的接口。
=======
**Bug 修复**

- 修复了 train api 中的 `_non_dist_train` ([#104](https://github.com/open-mmlab/mmediting/pull/104))
- 修复了 setup 和 CI ([#109](https://github.com/open-mmlab/mmediting/pull/109))
- 修复了 Normalize 中会导致多余循环的 bug ([#121](https://github.com/open-mmlab/mmediting/pull/121))
- 修复了 `get_hash` in `setup.py` ([#124](https://github.com/open-mmlab/mmediting/pull/124))
- 修复了 `tool/preprocess_reds_dataset.py` ([#148](https://github.com/open-mmlab/mmediting/pull/148))
- 修复了 `getting_started.md` 中的 slurm 训练教程 ([#162](https://github.com/open-mmlab/mmediting/pull/162))
- 修复了 pip 安装的 bug ([#173](https://github.com/open-mmlab/mmediting/pull/173))
- 修复了 config file 中的 bug ([#185](https://github.com/open-mmlab/mmediting/pull/185))
- 修复了数据集中失效的链接 ([#236](https://github.com/open-mmlab/mmediting/pull/236))
- 修复了 model zoo 中失效的链接 ([#242](https://github.com/open-mmlab/mmediting/pull/242))
>>>>>>> 6f2f3ae2ad3e365f94bbf19c01a1d1056dad3895

该版本存在有一些 BC-breaking 的修改。 请在[迁移指南](https://mmagic.readthedocs.io/zh_CN/latest/migration/overview.html)中查看更多细节。
