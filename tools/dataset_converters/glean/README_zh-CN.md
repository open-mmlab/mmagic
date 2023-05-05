# 准备 GLEAN 数据集

<!-- [DATASET] -->

```bibtex
@InProceedings{chan2021glean,
  author = {Chan, Kelvin CK and Wang, Xintao and Xu, Xiangyu and Gu, Jinwei and Loy, Chen Change},
  title = {GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

## 准备 cat_train 数据集

1. 从[LSUN 主页](https://www.yf.io/p/lsun)下载[cat 数据集](http://dl.yf.io/lsun/objects/cat.zip)。

2. 从[GLEAN 主页](https://github.com/ckkelvinchan/GLEAN)下载[cat_train/meta_info_LSUNcat_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/cat_train/meta_info_LSUNcat_GT.txt)。

3. 导出图像并下采样

从 lmdb 文件中导出图像，并下采样到所需尺寸。为此，我们提供了一个脚本：

```shell
python tools/dataset_converters/glean/preprocess_cat_train_dataset.py --lmdb-path .data/cat --meta-file-path ./data/cat_train/meta_info_LSUNcat_GT.txt --out-dir ./data/cat_train
```

生成的数据存储在 `cat_train` 目录下，目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── cat_train
│   │   ├── GT
│   │   ├── BIx8_down
│   │   ├── BIx16_down
│   │   ├── meta_info_LSUNcat_GT.txt
...
```

## 准备 cat_test 数据集

1. 从数据集[主页](https://archive.org/details/CAT_DATASET)下载[CAT 数据集](https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip)。

2. 从[GLEAN 主页](https://github.com/ckkelvinchan/GLEAN)下载[cat_test/meta_info_Cat100_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/cat_test/meta_info_Cat100_GT.txt)。

3. 下采样

将图像下采样到所需尺寸。为此，我们提供了一个脚本：

```shell
python tools/dataset_converters/glean/preprocess_cat_test_dataset.py --data-path ./data/CAT_03 --meta-file-path ./data/cat_test/meta_info_Cat100_GT.txt --out-dir ./data/cat_test
```

生成的数据存储在 `cat_test` 目录下，目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── cat_test
│   │   ├── GT
│   │   ├── BIx8_down
│   │   ├── BIx16_down
│   │   ├── meta_info_Cat100_GT.txt
...
```

## 准备 FFHQ 数据集

1. 从数据集[主页](https://github.com/NVlabs/ffhq-dataset)下载[FFHQ 数据集 (images1024x1024)](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)。

将文件目录重构为如下所示：

```text
ffhq
├── images
|   ├── 00000.png
|   ├── 00001.png
|   ├── ...
|   ├── 69999.png
```

2. 从[GLEAN 主页](https://github.com/ckkelvinchan/GLEAN)下载[ffhq/meta_info_FFHQ_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/FFHQ/meta_info_FFHQ_GT.txt)。

3. 下采样

将图像下采样到所需尺寸。为此，我们提供了一个脚本：

```shell
python tools/dataset_converters/glean/preprocess_ffhq_celebahq_dataset.py --data-root ./data/ffhq/images
```

生成的数据存储在 `ffhq` 目录下，目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── ffhq
|   |   ├── images
│   │   ├── BIx8_down
|   |   ├── BIx16_down
|   |   ├── meta_info_FFHQ_GT.txt
...
```

## 准备 CelebA-HQ 数据集

1. 根据数据集[主页](https://github.com/tkarras/progressive_growing_of_gans)文档准备数据集。

将文件目录重构为如下所示：

```text
CelebA-HQ
├── GT
|   ├── 00000.png
|   ├── 00001.png
|   ├── ...
|   ├── 30000.png
```

2. 从[GLEAN 主页](https://github.com/ckkelvinchan/GLEAN)下载[CelebA-HQ/meta_info_CelebAHQ_val100_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/CelebA-HQ/meta_info_CelebAHQ_val100_GT.txt)。

3. 下采样

将图像下采样到所需尺寸。为此，我们提供了一个脚本：

```shell
python tools/dataset_converters/glean/preprocess_ffhq_celebahq_dataset.py --data-root ./data/CelebA-HQ/GT
```

生成的数据存储在 `CelebA-HQ` 目录下，目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configsdata
├── data
|   ├── CelebA-HQ
|   |   ├── GT
│   │   ├── BIx8_down
|   |   ├── BIx16_down
|   |   ├── meta_info_CelebAHQ_val100_GT.txt
...
```

## 准备 FFHQ_CelebAHQ 数据集

将 FFHQ(`ffhq/images`) 和 CelebA-HQ(`CelebA-HQ/GT`) 合并，生成 FFHQ_CelebAHQ 数据集。

文件目录重构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── FFHQ_CelebAHQ
|   |   ├── GT
...
```
