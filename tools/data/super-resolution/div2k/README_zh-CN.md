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
- 验证集:  Set5 and Set14.

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │   ├── DIV2K_train_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   │   ├── DIV2K_valid_HR
│   │   ├── DIV2K_valid_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   ├── val_set5
│   │   ├── Set5_bicLRx2
│   │   ├── Set5_bicLRx3
│   │   ├── Set5_bicLRx4
│   ├── val_set14
│   │   ├── Set14_bicLRx2
│   │   ├── Set14_bicLRx3
│   │   ├── Set14_bicLRx4
```

## 裁剪子图

为了加快 IO，建议将 DIV2K 中的图片裁剪为一系列子图，为此，我们提供了一个脚本：

```shell
python tools/data/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

生成的数据保存在 `DIV2K` 目录下，其文件结构如下所示，其中 `_sub` 表示子图:

```text
mmediting
├── mmedit
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

## 准备 LMDB 格式的 DIV2K 数据集

如果您想使用 `LMDB` 以获得更快的 IO 速度，可以通过以下脚本来构建 LMDB 文件

```shell
python tools/data/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K --make-lmdb
```
