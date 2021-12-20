# 为 Pix2pix 准备配对数据集

<!-- [DATASET] -->

```bibtex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```

您可以从[此处](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)下载配对数据集。然后，您需要解压缩并移动相应的数据集以遵循如下所示的文件夹结构。数据集已经由原作者准备好了。

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── paired
│   │   ├── facades
│   │   ├── maps
|   |   ├── edges2shoes
|   |   |    ├── train
|   |   |    ├── test
```
