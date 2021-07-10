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

Vimeo90K 数据集包含了如下所示的`clip/sequence/img`目录结构：
```text
├── GT/LQ
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
```

## 准备Vimeo90K数据集的标注文件
为了准备好训练所需的标注文件，请先从Vimeo90K数据集官网下载训练路径列表，随后执行如下命令：

```shell
python tools/data/super-resolution/vimeo90k/preprocess_vimeo90k_dataset.py ./data/Vimeo90K/official_train_list.txt
```

测试集的标注文件可通过类似方式生成.


## 准备LMDB格式的Vimeo90K数据集

如果你想使用`LMDB`以获得更快的IO速度，可以通过以下脚本来构建LMDB文件
```shell
python tools/data/super-resolution/vimeo90k/preprocess_vimeo90k_dataset.py ./data/Vimeo90K/official_train_list.txt --gt_path ./data/Vimeo90K/GT --lq_path ./data/Vimeo90K/LQ  --make-lmdb
```
