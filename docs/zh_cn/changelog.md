# 变更日志

## v1.0.0rc7 (06/04/2023)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc7 版本。 此版本支持了 MMEditing 和 MMGeneration 的 50+ 模型，222+ configs 和 209+ checkpoints。以下是此次版本发布的重点新功能

- 支持了 Inpainting 任务推理的 Gradio gui.
- 支持了图像上色、图像翻译和 GAN 模型的 inferencer.

**新功能和改进**

- 支持了图像超分和视频超分的 inferencer. [#1662](https://github.com/open-mmlab/mmediting/pull/1662), [#1720](https://github.com/open-mmlab/mmediting/pull/1720)
- 重构 get_flops 脚本. [#1675](https://github.com/open-mmlab/mmediting/pull/1675)
- 重构数据集的 dataset_converters 脚本和使用文档. [#1690](https://github.com/open-mmlab/mmediting/pull/1690)
- 迁移 stylegan 算子到 MMCV 中. [#1383](https://github.com/open-mmlab/mmediting/pull/1383)

**Bug 修复**

- 修复 disco inferencer. [#1673](https://github.com/open-mmlab/mmediting/pull/1673)
- 修复 nafnet optimizer 配置. [#1716](https://github.com/open-mmlab/mmediting/pull/1716)
- 修复 tof typo. [#1711](https://github.com/open-mmlab/mmediting/pull/1711)

**贡献者**

@LeoXing1996, @Z-Fran, @plyfager, @zengyh1900, @liuwenran, @ryanxingql, @HAOCHENYE, @VongolaWu

## v1.0.0rc6 (02/03/2023)

**主要更新**

我们很高兴发布 MMEditing 1.0.0rc6 版本。 此版本支持了 MMEditing 和 MMGeneration 的 50+ 模型，222+ configs 和 209+ checkpoints。以下是此次版本发布的重点新功能

- 支持了 Inpainting 任务推理的 Gradio gui.
- 支持了图像上色、图像翻译和 GAN 模型的 inferencer.

**新功能和改进**

- 重构了 FileIO. [#1572](https://github.com/open-mmlab/mmediting/pull/1572)
- 重构了 registry. [#1621](https://github.com/open-mmlab/mmediting/pull/1621)
- 重构了 Random degradations. [#1583](https://github.com/open-mmlab/mmediting/pull/1583)
- 重构了 DataSample, DataPreprocessor, Metric 和 Loop. [#1656](https://github.com/open-mmlab/mmediting/pull/1656)
- 使用 mmengine.basemodule 替换 nn.module. [#1491](https://github.com/open-mmlab/mmediting/pull/1491)
- 重构了算法库主页. [#1609](https://github.com/open-mmlab/mmediting/pull/1609)
- 支持了 Inpainting 任务推理的 Gradio gui. [#1601](https://github.com/open-mmlab/mmediting/pull/1601)
- 支持了图像上色的 inferencer. [#1588](https://github.com/open-mmlab/mmediting/pull/1588)
- 支持了图像翻译和所有 GAN 模型的 inferencer. [#1650](https://github.com/open-mmlab/mmediting/pull/1650)
- 支持了 GAN 模型的 inferencer. [#1653](https://github.com/open-mmlab/mmediting/pull/1653), [#1659](https://github.com/open-mmlab/mmediting/pull/1659)
- 新增 Print config 工具. [#1590](https://github.com/open-mmlab/mmediting/pull/1590)
- 改进 type hints. [#1604](https://github.com/open-mmlab/mmediting/pull/1604)
- 更新 metrics 和 datasets 的中文文档. [#1568](https://github.com/open-mmlab/mmediting/pull/1568), [#1638](https://github.com/open-mmlab/mmediting/pull/1638)
- 更新 BigGAN 和 Disco-Diffusion 的中文文档. [#1620](https://github.com/open-mmlab/mmediting/pull/1620)
- 更新 Guided-Diffusion 的 Evaluation 和 README. [#1547](https://github.com/open-mmlab/mmediting/pull/1547)

**Bug 修复**

- 修复 EMA `momentum`. [#1581](https://github.com/open-mmlab/mmediting/pull/1581)
- 修复 RandomNoise 的输出类型. [#1585](https://github.com/open-mmlab/mmediting/pull/1585)
- 修复 pytorch2onnx 工具. [#1629](https://github.com/open-mmlab/mmediting/pull/1629)
- 修复 API 文档. [#1641](https://github.com/open-mmlab/mmediting/pull/1641), [#1642](https://github.com/open-mmlab/mmediting/pull/1642)
- 修复 RealESRGAN 加载 EMA 参数. [#1647](https://github.com/open-mmlab/mmediting/pull/1647)
- 修复 dataset_converters 脚本的 arg passing bug. [#1648](https://github.com/open-mmlab/mmediting/pull/1648)

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

该版本存在有一些 BC-breaking 的修改。 请在[迁移指南](https://mmediting.readthedocs.io/zh_CN/latest/migration/overview.html)中查看更多细节。
