# 教程 1: 了解配置文件

mmedit 采用基于 python 文件的配置系统，您可以在 `$MMEditing/configs` 下查看预置的配置文件。

## 配置文件命名风格

配置文件按照下面的风格命名。我们建议社区贡献者使用同样的风格。

```bash
{model}_[model setting]_{backbone}_[refiner]_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

`{xxx}` 是必填字段，`[yyy]` 是可选的。

- `{model}`: 模型种类，例如 `srcnn`, `dim` 等等。
- `[model setting]`: 特定设置一些模型，例如，输入图像 `resolution` , 训练 `stage name`。
- `{backbone}`: 主干网络种类，例如 `r50` (ResNet-50)、`x101` (ResNeXt-101)。
- `{refiner}`: 精炼器种类，例如 `pln` 简单精炼器模型
- `[norm_setting]`: 指定归一化设置，默认为批归一化，其他归一化可以设为: `bn`(批归一化), `gn` (组归一化), `syncbn` (同步批归一化)。
- `[misc]`: 模型中各式各样的设置/插件，例如 `dconv`, `gcb`, `attention`, `mstrain`。
- `[gpu x batch_per_gpu]`: GPU数目 和每个 GPU 的样本数， 默认为 `8x2 `。
- `{schedule}`: 训练策略，如 `20k`, `100k` 等，意思是 `20k` 或 `100k` 迭代轮数。
- `{dataset}`: 数据集，如 `places`（图像补全）、`comp1k`（抠图）、`div2k`（图像恢复）和 `paired`（图像生成）。
