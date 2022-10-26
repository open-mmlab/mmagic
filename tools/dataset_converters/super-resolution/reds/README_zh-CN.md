# 准备 REDS 数据集

<!-- [DATASET] -->

```bibtex
@InProceedings{Nah_2019_CVPR_Workshops_REDS,
  author = {Nah, Seungjun and Baik, Sungyong and Hong, Seokil and Moon, Gyeongsik and Son, Sanghyun and Timofte, Radu and Lee, Kyoung Mu},
  title = {NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2019}
}
```

- 训练集: [REDS 数据集](https://seungjunnah.github.io/Datasets/reds.html).
- 验证集: [REDS 数据集](https://seungjunnah.github.io/Datasets/reds.html) 和 Vid4.

请注意，我们合并了 REDS 的训练集和验证集，以便在 REDS4 划分（在 `EDVR` 中会使用到）和官方验证集划分之间切换。

原始验证集的名称被修改了（clip 000 到 029），以避免与训练集发生冲突（总共 240 个 clip）。具体而言，验证集中的 clips 被改名为 240、241、... 269。

可通过运行以下命令来准备 REDS 数据集:

```shell
python tools/data/super-resolution/reds/preprocess_reds_dataset.py ./data/REDS
```

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── REDS
│   │   ├── train_sharp
│   │   │   ├── 000
│   │   │   ├── 001
│   │   │   ├── ...
│   │   ├── train_sharp_bicubic
│   │   │   ├── 000
│   │   │   ├── 001
│   │   │   ├── ...
│   ├── REDS4
│   │   ├── GT
│   │   ├── sharp_bicubic
```

## 准备 LMDB 格式的 REDS 数据集

如果您想使用 `LMDB` 以获得更快的 IO 速度，可以通过以下脚本来构建 LMDB 文件：

```shell
python tools/data/super-resolution/reds/preprocess_reds_dataset.py --root-path ./data/REDS --make-lmdb
```

## 裁剪为子图

MMEditing 支持将 REDS 图像裁剪为子图像以加快 IO。我们提供了这样一个脚本：

```shell
python tools/data/super-resolution/reds/crop_sub_images.py --data-root ./data/REDS  -scales 4
```

生成的数据存储在 `REDS` 下，数据结构如下，其中`_sub`表示子图像。

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── REDS
│   │   ├── train_sharp
│   │   │   ├── 000
│   │   │   ├── 001
│   │   │   ├── ...
│   │   ├── train_sharp_sub
│   │   │   ├── 000_s001
│   │   │   ├── 000_s002
│   │   │   ├── ...
│   │   │   ├── 001_s001
│   │   │   ├── ...
│   │   ├── train_sharp_bicubic
│   │   │   ├── X4
│   │   │   │   ├── 000
│   │   │   │   ├── 001
│   │   │   │   ├── ...
│   │   │   ├── X4_sub
│   │   │   ├── 000_s001
│   │   │   ├── 000_s002
│   │   │   ├── ...
│   │   │   ├── 001_s001
│   │   │   ├── ...
```

请注意，默认情况下，`preprocess_reds_dataset.py` 不会为裁剪后的数据集制作 lmdb 和注释文件。您可能需要为此类操作稍微修改脚本。
