# Preparing Vid4 Dataset

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

The Vid4 dataset can be downloaded from [here](https://drive.google.com/file/d/1ZuvNNLgR85TV_whJoHM7uVb-XW1y70DW/view?usp=sharing). There are two degradations in the dataset.

1. BIx4 contains images downsampled by bicubic interpolation
2. BDx4 contains images blurred by Gaussian kernel with σ=1.6, followed by a subsampling every four pixels.

Note that we should prepare a annotation file (such as meta_info_Vid4_GT.txt) for Vid4 dataset as follows.

```text
calendar 41 (576,720,3)
city 34 (576,704,3)
foliage 49 (480,720,3)
walk 47 (480,720,3)
```

For ToFlow, we should prepare directly upsampling dataset. We provide such a script:

```shell
python tools/dataset_converters/vid4/preprocess_vid4_dataset.py --data-root ./data/Vid4/BIx4
```

The folder structure should look like:

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
