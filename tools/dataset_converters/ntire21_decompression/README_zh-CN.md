# 准备 NTIRE21 decompression 数据集

<!-- [DATASET] -->

```bibtex
@inproceedings{yang2021dataset,
  title={{NTIRE 2021} Challenge on Quality Enhancement of Compressed Video: Dataset and Study},
  author={Ren Yang and Radu Timofte},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
```

测试数据集可以从其[主页](https://github.com/RenYang-home/NTIRE21_VEnh)下载。

请按照[主页](https://github.com/RenYang-home/NTIRE21_VEnh)教程生成数据集。

文件目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── NTIRE21_decompression_track1
|   |   ├── GT
|   |   |   ├── 001
|   |   |   |   ├── 001.png
|   |   |   |   ├── ...
|   |   |   ├── ...
|   |   |   ├── 010
|   |   ├── LQ
|   |   |   ├── 001
|   |   |   |   ├── 001.png
|   |   |   |   ├── ...
|   |   |   ├── ...
|   |   |   ├── 010
|   ├── NTIRE21_decompression_track2
|   |   ├── GT
|   |   ├── LQ
|   ├── NTIRE21_decompression_track3
|   |   ├── GT
|   |   ├── LQ
```
