# 准备 Vimeo90K 数据集

<!-- [DATASET] -->

```bibtex
@article{xue2019video,
  title={Video Enhancement with Task-Oriented Flow},
  author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  journal={International Journal of Computer Vision (IJCV)},
  volume={127},
  number={8},
  pages={1106--1125},
  year={2019},
  publisher={Springer}
}
```

训练集和测试集可以从 [此处](http://toflow.csail.mit.edu/) 下载。

将数据集路径 `vimeo_septuplet/sequences` 重命名为 `vimeo90k/GT`。Vimeo90K 数据集包含了如下所示的 `clip/sequence/img` 目录结构：

```text
vimeo90k
├── GT
│   ├── 00001
│   │   ├── 0001
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   ├── ...
│   │   ├── 0002
│   │   ├── 0003
│   │   ├── ...
│   ├── 00002
│   ├── ...
├── sep_trainlist.txt
├── sep_testlist.txt
```

为了生成下采样图像BIx4和BDx4，以及准备所需的标注文件，需要执行如下命令：

```shell
python tools/dataset_converters/vimeo90k/preprocess_vimeo90k_dataset.py --data-root ./data/vimeo90k
```

文件目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── vimeo_triplet
│   │   ├── GT
│   │   │   ├── 00001
│   │   │   │   ├── 0001
│   │   │   │   │   ├── im1.png
│   │   │   │   │   ├── im2.png
│   │   │   │   │   ├── ...
│   │   │   │   ├── 0002
│   │   │   │   ├── 0003
│   │   │   │   ├── ...
│   │   │   ├── 00002
│   │   │   ├── ...
│   │   ├── BIx4
│   │   ├── BDx4
│   │   ├── meta_info_Vimeo90K_test_GT.txt
│   │   ├── meta_info_Vimeo90K_train_GT.txt
```

## 准备 LMDB 格式的 Vimeo90K 数据集

如果您想使用 `LMDB` 以获得更快的 IO 速度，可以通过以下脚本来构建 LMDB 文件

```shell
python tools/dataset_converters/vimeo90k/preprocess_vimeo90k_dataset.py --data-root ./data/vimeo90k --train_list ./data/vimeo90k/sep_trainlist.txt --gt-path ./data/vimeo90k/GT --lq-path ./data/Vimeo90k/BIx4  --make-lmdb
```
