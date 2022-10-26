# Preparing REDS Dataset

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

## Prepare LMDB dataset for REDS

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/data/super-resolution/reds/preprocess_reds_dataset.py --root-path ./data/REDS --make-lmdb
```

## Crop to sub-images

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
