# Super-Resolution Datasets

It is recommended to symlink the dataset root to `$MMEDITING/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

MMEditing supported super-resolution datasets:

- Image Super-Resolution
  - [DF2K_OST](#df2kost-dataset) \[ [Homepage](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md) \]
  - [DIV2K](#div2k-dataset) \[ [Homepage](https://data.vision.ee.ethz.ch/cvl/DIV2K/) \]
- Video Super-Resolution
  - [REDS](#reds-dataset) \[ [Homepage](https://seungjunnah.github.io/Datasets/reds.html) \]
  - [Vimeo90K](#vimeo90k-dataset) \[ [Homepage](http://toflow.csail.mit.edu) \]

## DF2K_OST Dataset

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
mmediting
├── mmedit
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

### Crop sub-images

For faster IO, we recommend to crop the images to sub-images. We provide such a script:

```shell
python tools/data/super-resolution/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost
```

The generated data is stored under `df2k_ost` and the data structure is as follows, where `_sub` indicates the sub-images.

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── df2k_ost
│   │   ├── GT
│   │   ├── GT_sub
...
```

### Prepare LMDB dataset for DF2K_OST

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/data/super-resolution/df2k_ost/preprocess_df2k_ost_dataset.py --data-root ./data/df2k_ost --make-lmdb
```

## DIV2K Dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{Agustsson_2017_CVPR_Workshops,
    author = {Agustsson, Eirikur and Timofte, Radu},
    title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {July},
    year = {2017}
}
```

- Training dataset:  [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Validation dataset:  Set5 and Set14.

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │   ├── DIV2K_train_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   │   ├── DIV2K_valid_HR
│   │   ├── DIV2K_valid_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   ├── Set5
│   │   ├── GTmod12
│   │   ├── LRbicx2
│   │   ├── LRbicx3
│   │   ├── LRbicx4
│   ├── Set14
│   │   ├── GTmod12
│   │   ├── LRbicx2
│   │   ├── LRbicx3
│   │   ├── LRbicx4
```

### Crop sub-images

For faster IO, we recommend to crop the DIV2K images to sub-images. We provide such a script:

```shell
python tools/data/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

The generated data is stored under `DIV2K` and the data structure is as follows, where `_sub` indicates the sub-images.

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │   ├── DIV2K_train_HR_sub
│   │   ├── DIV2K_train_LR_bicubic
│   │   │   ├── X2
│   │   │   ├── X3
│   │   │   ├── X4
│   │   │   ├── X2_sub
│   │   │   ├── X3_sub
│   │   │   ├── X4_sub
│   │   ├── DIV2K_valid_HR
│   │   ├── ...
...
```

### Prepare annotation list

If you use the annotation mode for the dataset, you first need to prepare a specific `txt` file.

Each line in the annotation file contains the image names and image shape (usually for the ground-truth images), separated by a white space.

Example of an annotation file:

```text
0001_s001.png (480,480,3)
0001_s002.png (480,480,3)
```

### Prepare LMDB dataset for DIV2K

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/data/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K --make-lmdb
```

## REDS Dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{Nah_2019_CVPR_Workshops_REDS,
  author = {Nah, Seungjun and Baik, Sungyong and Hong, Seokil and Moon, Gyeongsik and Son, Sanghyun and Timofte, Radu and Lee, Kyoung Mu},
  title = {NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2019}
}
```

- Training dataset: [REDS dataset](https://seungjunnah.github.io/Datasets/reds.html).
- Validation dataset: [REDS dataset](https://seungjunnah.github.io/Datasets/reds.html) and Vid4.

Note that we merge train and val datasets in REDS for easy switching between REDS4 partition (used in EDVR) and the official validation partition.
The original val dataset (clip names from 000 to 029) are modified to avoid conflicts with training dataset (total 240 clips). Specifically, the clip names are changed to 240, 241, ... 269.

You can prepare the REDS dataset by running:

```shell
python tools/data/super-resolution/reds/preprocess_reds_dataset.py --root-path ./data/REDS
```

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── REDS
│   │   ├── train_sharp
│   │   │   ├── 000
│   │   │   ├── 001
│   │   │   ├── ...
│   │   ├── train_sharp_bicubic
│   │   │   ├── 000
│   │   │   ├── 001
│   │   │   ├── ...
│   ├── REDS4
│   │   ├── GT
│   │   ├── sharp_bicubic
```

### Prepare LMDB dataset for REDS

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/data/super-resolution/reds/preprocess_reds_dataset.py --root-path ./data/REDS --make-lmdb
```

### Crop to sub-images

MMEditing also support cropping REDS images to sub-images for faster IO. We provide such a script:

```shell
python tools/data/super-resolution/reds/crop_sub_images.py --data-root ./data/REDS  -scales 4
```

The generated data is stored under `REDS` and the data structure is as follows, where `_sub` indicates the sub-images.

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── REDS
│   │   ├── train_sharp
│   │   │   ├── 000
│   │   │   ├── 001
│   │   │   ├── ...
│   │   ├── train_sharp_sub
│   │   │   ├── 000_s001
│   │   │   ├── 000_s002
│   │   │   ├── ...
│   │   │   ├── 001_s001
│   │   │   ├── ...
│   │   ├── train_sharp_bicubic
│   │   │   ├── X4
│   │   │   │   ├── 000
│   │   │   │   ├── 001
│   │   │   │   ├── ...
│   │   │   ├── X4_sub
│   │   │   ├── 000_s001
│   │   │   ├── 000_s002
│   │   │   ├── ...
│   │   │   ├── 001_s001
│   │   │   ├── ...
```

Note that by default `preprocess_reds_dataset.py` does not make lmdb and annotation file for the cropped dataset. You may need to modify the scripts a little bit for such operations.

## Vid4 Dataset

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

## Vimeo90K Dataset

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

The training and test datasets can be download from [here](http://toflow.csail.mit.edu/).

The Vimeo90K dataset has a `clip/sequence/img` folder structure:

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── vimeo_triplet
│   │   ├── BDx4
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
│   │   ├── GT
│   │   ├── meta_info_Vimeo90K_test_GT.txt
│   │   ├── meta_info_Vimeo90K_train_GT.txt
```

### Prepare the annotation files for Vimeo90K dataset

To prepare the annotation file for training, you need to download the official training list path for Vimeo90K from the official website, and run the following command:

```shell
python tools/data/super-resolution/vimeo90k/preprocess_vimeo90k_dataset.py ./data/Vimeo90K/official_train_list.txt
```

The annotation file for test is generated similarly.

### Prepare LMDB dataset for Vimeo90K

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/data/super-resolution/vimeo90k/preprocess_vimeo90k_dataset.py ./data/Vimeo90K/official_train_list.txt --gt-path ./data/Vimeo90K/GT --lq-path ./data/Vimeo90K/LQ  --make-lmdb
```
