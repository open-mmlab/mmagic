# Preparing DIV2K Dataset

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
│   ├── val_set5
│   │   ├── Set5_bicLRx2
│   │   ├── Set5_bicLRx3
│   │   ├── Set5_bicLRx4
│   ├── val_set14
│   │   ├── Set14_bicLRx2
│   │   ├── Set14_bicLRx3
│   │   ├── Set14_bicLRx4
```

## Crop sub-images

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

## Prepare annotation list

If you use the annotation mode for the dataset, you first need to prepare a specific `txt` file.

Each line in the annotation file contains the image names and image shape (usually for the ground-truth images), separated by a white space.

Example of an annotation file:

```text
0001_s001.png (480,480,3)
0001_s002.png (480,480,3)
```

## Prepare LMDB dataset for DIV2K

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/data/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K --make-lmdb
```
