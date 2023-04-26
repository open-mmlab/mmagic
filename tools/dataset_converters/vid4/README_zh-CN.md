# 准备 Vid4 数据集

<!-- [DATASET] -->

```bibtex
@article{xue2019video,
  title={On Bayesian adaptive video super resolution},
  author={Liu, Ce and Sun, Deqing},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={36},
  number={2},
  pages={346--360},
  year={2013},
  publisher={IEEE}
}
```

可以从 [此处](https://drive.google.com/file/d/1ZuvNNLgR85TV_whJoHM7uVb-XW1y70DW/view?usp=sharing) 下载 Vid4 数据集，其中包含了由两种下采样方法得到的图片：

1. BIx4 包含了由双线性插值下采样得到的图片
2. BDx4 包含了由 `σ=1.6` 的高斯核模糊，然后每4个像素进行一次采样得到的图片

请注意，应为 Vid4 数据集准备一个如下所列的标注文件（例如 meta_info_Vid4_GT.txt）。

```text
calendar 41 (576,720,3)
city 34 (576,704,3)
foliage 49 (480,720,3)
walk 47 (480,720,3)
```

对于 ToFlow，应准备直接上采样的数据集，为此，我们提供了一个脚本：

```shell
python tools/dataset_converters/vid4/preprocess_vid4_dataset.py --data-root ./data/Vid4/BIx4
```

文件目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── Vid4
│   │   ├── GT
│   │   │   ├── calendar
│   │   │   ├── city
│   │   │   ├── foliage
│   │   │   ├── walk
│   │   ├── BDx4
│   │   ├── BIx4
│   │   ├── BIx4up_direct
│   │   ├── meta_info_Vid4_GT.txt
```
