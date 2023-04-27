# Preparing DF2K_OST Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{wang2021real,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1905--1914},
  year={2021}
}
```

- The DIV2K dataset can be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (We use the training set only).
- The Flickr2K dataset can be downloaded [here](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (We use the training set only).
- The OST dataset can be downloaded [here](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip) (We use the training set only).

Please first put all the images into the `GT` folder (naming does not need to be in order):

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── df2k_ost
│   │   ├── GT
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   ├── ...
...
```

## Crop sub-images

For faster IO, we recommend to crop the images to sub-images. We provide such a script:

```shell
python tools/dataset_converters/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost
```

The generated data is stored under `df2k_ost` and the data structure is as follows, where `_sub` indicates the sub-images.

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── df2k_ost
│   │   ├── GT
│   │   ├── GT_sub
│   │   ├── meta_info_df2k_ost.txt
...
```

## Prepare annotation list

If you use the annotation mode for the dataset, you first need to prepare a specific `txt` file.

Each line in the annotation file contains the image names and image shape (usually for the ground-truth images), separated by a white space.

Example of an annotation file:

```text
0001_s001.png (480,480,3)
0001_s002.png (480,480,3)
```

Note that `preprocess_df2k_ost_dataset.py` will generate default annotation files.

## Prepare LMDB dataset for DF2K_OST

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/dataset_converters/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost --make-lmdb
```
