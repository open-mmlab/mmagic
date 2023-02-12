# 教程 2：准备数据集

在这一章节，我们将会详细的讨论如何准备数据集和在仓库中如何针对不同的模型采用合适的数据集。

我们为不同的任务准备了多个数据集。

大家可以通过以下两个步骤在MMEditing中使用数据集完成训练和测试的工作：

1. 直接使用下载的数据集。
2. 在使用之前对下载好的数据集进行预处理操作。

教程的指引分为以下几个部分

- [教程 2：准备数据集](#教程 2：准备数据集)
  - [下载数据集](#下载数据集)
  - [准备数据集](#准备数据集)
  - [MMEditing中的数据集概览](#MMEditing中的数据集概览)

## 下载数据集

大家应该先从他们的主页上下载数据集

大部分数据集下载后即可使用，只需确保文件夹结构正确即可，无需额外准备。

例如，您可以通过从 [主页](http://toflow.csail.mit.edu/) 下载数据集来简单地准备 Vimeo90K-triplet 数据集。

## 准备数据集

一些数据集需要在训练或测试之前进行预处理。我们支持许多脚本来准备 [tools/dataset_converters](https://github.com/open-mmlab/mmediting/tree/1.x/tools/dataset_converters) 中的数据集。您可以按照每个数据集的教程来运行脚本。

例如，我们建议将 DIV2K 图像裁剪为子图像。我们提供了一个脚本来准备裁剪后的 DIV2K 数据集。您可以运行以下命令：

```shell
python tools/dataset_converters/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

## MMEditing 中的数据集概览

我们支持详细的教程，并根据不同的任务进行拆分。

请查看我们的 dataset zoo，了解不同任务的数据准备。

如果您想了解更多关于 MMEditing 中数据集的详细信息，欢迎查看 [高级指南](../howto/dataset.md)。
