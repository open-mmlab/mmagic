# 准备 DIV2K 数据集

<!-- [DATASET] -->

```bibtex
@InProceedings{Agustsson_2017_CVPR_Workshops,
    author = {Agustsson, Eirikur and Timofte, Radu},
    title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {July},
    year = {2017}
}
```

- 训练集:  [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- 验证集:  [Set5](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) 和 [Set14](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u).

请注意，我们将原始的验证集（文件名 0801 到 0900）合并进了原始的训练集（文件名 0001 到 0800）。文件目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   ├── ...
│   │   │   ├── 0800.png
│   │   │   ├── 0801.png
│   │   │   ├── ...
│   │   │   ├── 0900.png
│   │   ├── DIV2K_train_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   │   ├── DIV2K_valid_HR
│   │   ├── DIV2K_valid_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   ├── Set5
│   │   ├── GTmod12
│   │   ├── LRbicx2
│   │   ├── LRbicx3
│   │   ├── LRbicx4
│   ├── Set14
│   │   ├── GTmod12
│   │   ├── LRbicx2
│   │   ├── LRbicx3
│   │   ├── LRbicx4
```

## 裁剪子图

为了加快 IO，建议将 DIV2K 中的图片裁剪为一系列子图，为此，我们提供了一个脚本：

```shell
python tools/dataset_converters/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

生成的数据保存在 `DIV2K` 目录下，其文件结构如下所示，其中 `_sub` 表示子图:

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │   ├── DIV2K_train_HR_sub
│   │   ├── DIV2K_train_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   │   │   ├── X2_sub
│   │   │   ├── X3_sub
│   │   │   ├── X4_sub
│   │   ├── DIV2K_valid_HR
│   │   ├── ...
│   │   ├── meta_info_DIV2K800sub_GT.txt
│   │   ├── meta_info_DIV2K100sub_GT.txt
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

请注意，`preprocess_div2k_dataset` 脚本默认生成一份标注文件。

## 准备 LMDB 格式的 DIV2K 数据集

如果您想使用 `LMDB` 以获得更快的 IO 速度，可以通过以下脚本来构建 LMDB 文件

```shell
python tools/dataset_converters/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K --make-lmdb
```
