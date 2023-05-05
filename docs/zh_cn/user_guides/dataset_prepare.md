# 教程 2：准备数据集

在本节中，我们将详细介绍如何准备数据并在本仓库的不同任务中采用适当的数据集。

我们支持不同任务的多个数据集。
在MMagic中，有两种方法可以将数据集用于训练和测试模型：

1. 直接使用下载的数据集
2. 在使用下载的数据集之前对其进行预处理

本文的结构如下：

- \[教程 2：准备数据集\](#教程 2：准备数据集)
  - [下载数据集](#下载数据集)
  - [准备数据集](#准备数据集)
  - [MMagic中的数据集概述](#MMagic中的数据集概述)

## 下载数据集

首先，建议您从官方的页面下载数据集。
大多数数据集在下载后都是可用的，因此您只需确保文件夹结构正确，无需进一步准备。
例如，您可以通过从[主页](http://toflow.csail.mit.edu/)下载，来简单地准备Vimeo90K-triplet数据集.

## 准备数据集

一些数据集需要在训练或测试之前进行预处理。我们在
[tools/dataset_converters](https://github.com/open-mmlab/mmagic/tree/main/tools/dataset_converters)中支持许多用来准备数据集的脚本。
您可以遵循每个数据集的教程来运行脚本。例如，我们建议将DIV2K图像裁剪为子图像。我们提供了一个脚本来准备裁剪的DIV2K数据集。可以运行以下命令：

```shell
python tools/dataset_converters/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

## MMagic中的数据集概述

我们支持详细的教程，并根据不同的任务进行拆分。

请查看我们的数据集概览，了解不同任务的数据准备。

如果您对MMagic中数据集的更多细节感兴趣，请查看[进阶教程](../howto/dataset.md)。
