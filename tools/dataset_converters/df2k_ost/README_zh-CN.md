# 准备 DF2K_OST 数据集

<!-- [DATASET] -->

```bibtex
@inproceedings{wang2021real,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1905--1914},
  year={2021}
}
```

- DIV2K 数据集可以在 [这里](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 下载 (我们只使用训练集)。
- Flickr2K 数据集可以在 [这里](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) 下载 (我们只使用训练集)。
- OST 数据集可以在 [这里](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip) 下载 (我们只使用训练集)。

请先将所有图片放入 `GT` 文件夹（命名不需要按顺序）：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── df2k_ost
│   │   ├── GT
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   ├── ...
...
```

## 裁剪子图像

为了更快的 IO，我们建议将图像裁剪为子图像。 我们提供了这样一个脚本：

```shell
python tools/dataset_converters/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost
```

生成的数据存放在 `df2k_ost` 下，数据结构如下，其中 `_sub` 表示子图像。

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── df2k_ost
│   │   ├── GT
│   │   ├── GT_sub
│   │   ├── meta_info_df2k_ost.txt
...
```

## 准备标注列表文件

如果您想使用`标注模式`来处理数据集，需要先准备一个 `txt` 格式的标注文件。

标注文件中的每一行包含了图片名以及图片尺寸（这些通常是 ground-truth 图片），这两个字段用空格间隔开。

标注文件示例:

```text
0001_s001.png (480,480,3)
0001_s002.png (480,480,3)
```

请注意，`preprocess_df2k_ost_dataset.py` 脚本默认生成一份标注文件。

## Prepare LMDB dataset for DF2K_OST

如果你想使用 LMDB 数据集来获得更快的 IO 速度，你可以通过以下方式制作 LMDB 文件：

```shell
python tools/dataset_converters/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost --make-lmdb
```
