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
- OST 数据集可以在 [这里](https://github.com/xinntao/SFTGAN#ost-dataset) 下载 (我们只使用训练集 OutdoorSceneTrain_v2 )。

请先将所有图片放入 `GT` 文件夹（命名不需要按顺序）：

```text
mmediting
├── mmedit
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
python tools/data/super-resolution/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost
```

生成的数据存放在 `df2k_ost` 下，数据结构如下，其中 `_sub` 表示子图像。

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── df2k_ost
│   │   ├── GT
│   │   ├── GT_sub
...
```

## Prepare LMDB dataset for DF2K_OST

如果你想使用 LMDB 数据集来获得更快的 IO 速度，你可以通过以下方式制作 LMDB 文件：

```shell
python tools/data/super-resolution/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost --make-lmdb
```
