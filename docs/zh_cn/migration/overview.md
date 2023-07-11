# 概览

本节将从以下几个方面介绍如何从 MMEditing 0.x 迁移至 MMagic 1.x：

- [概览](#概览)
  - [新依赖项](#新依赖项)
  - [总体结构](#总体结构)
  - [其他配置设置](#其他配置设置)

## 新依赖项

MMagic 1.x 依赖于一些新的包，您可以按照[安装教程](../get_started/install.md)准备一个新的干净环境并重新安装。

## 总体结构

我们在 MMagic 1.x 中对总体结构进行了重构，具体如下：

- 旧版本 MMEdit 中的 `core` 被拆分为 `engine`、`evaluation`、`structures` 和 `visualization`
- 旧版本 MMEdit 中 `datasets` 的 `pipelines` 被重构为 `transforms`
- MMagic 1.x 中的 `models` 被重构为六个部分：`archs`、`base_models`、`data_preprocessors`、`editors`、`diffusion_schedulers` 和 `losses`。

## 其他配置设置

我们将配置文件重命名为新模板：`{model_settings}_{module_setting}_{training_setting}_{datasets_info}`。

更多配置细节请参见[配置指南](../user_guides/config.md)。
