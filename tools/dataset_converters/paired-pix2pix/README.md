# Preparing Paired Dataset for Pix2pix

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

You can download paired datasets from [here](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).
Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.

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
