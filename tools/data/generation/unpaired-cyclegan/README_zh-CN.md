# 为 CycleGAN 准备未配对数据集

<!-- [DATASET] -->

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}
```

您可以从[此处](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)下载未配对的数据集。然后，您需要解压缩并移动相应的数据集以遵循如上所示的文件夹结构。数据集已经由原作者准备好了。

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── unpaired
│   │   ├── facades
|   |   ├── horse2zebra
|   |   ├── summer2winter_yosemite
|   |   |    ├── trainA
|   |   |    ├── trainB
|   |   |    ├── testA
|   |   |    ├── testB
```
