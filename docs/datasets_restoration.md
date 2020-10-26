## Prepare Datasets for Restoration

### Image Super-Resolution

- Training dataset:  [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Validation dataset:  Set5 and Set14.

It is recommended to symlink the dataset root to `$MMEditing/data`:

```
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

If your folder structure is different, you may need to change the corresponding paths in config files.

#### Crop sub-images
For faster IO, we recommend to crop the DIV2K images to sub-images. We provide such a script:
```shell
python tools/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

The generated data is stored under `DIV2K` and the data structure is as follows, where `_sub` indicates the sub-images.
```
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

#### Prepare annotation list
If you use the annotation mode for the dataset, you first need to prepare a specific `txt` file.

Each line in the annotation file contains the image names and image shape (usually for the ground-truth images), separated by a white space.

Example of an annotation file:
```
0001_s001.png (480,480,3)
0001_s002.png (480,480,3)
```

### Video Super-Resolution
- Training dataset: [REDS dataset](https://seungjunnah.github.io/Datasets/reds.html).
- Validation dataset: [REDS dataset](https://seungjunnah.github.io/Datasets/reds.html) and Vid4.

Note that we merge train and val datasets in REDS for easy switching between REDS4 partition (used in EDVR) and the official validation partition. <br>
The original val dataset (clip names from 000 to 029) are modified to avoid conflicts with training dataset (total 240 clips). Specifically, the clip names are changed to 240, 241, ... 269.

You can prepare the REDS dataset by running:
```shell
python tools/preprocess_reds_dataset.py ./data/REDS
```

It is also recommended to symlink the dataset root to `$MMEditing/data`:

```
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

### Prepare LMDB dataset
If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:
```shell
# DIV2K
python tools/preprocess_div2k_dataset.py --data-root ./data/DIV2K --make-lmdb
# REDS
python tools/preprocess_reds_dataset.py --root-path ./data/REDS --make-lmdb
```
