# Preparing Unpaired Dataset for CycleGAN

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

You can download unpaired datasets from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).
Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.

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
