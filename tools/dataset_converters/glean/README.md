# Preparing GLEAN Dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{chan2021glean,
  author = {Chan, Kelvin CK and Wang, Xintao and Xu, Xiangyu and Gu, Jinwei and Loy, Chen Change},
  title = {GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

## Preparing cat_train dataset

1. Download [cat dataset](http://dl.yf.io/lsun/objects/cat.zip) from [LSUN homepage](https://www.yf.io/p/lsun)

2. Download [cat_train/meta_info_LSUNcat_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/cat_train/meta_info_LSUNcat_GT.txt) from [GLEAN homepage](https://github.com/ckkelvinchan/GLEAN)

3. Export and downsample images

Export images from lmdb file and resize the input images to the designated size. We provide such a script:

```shell
python tools/dataset_converters/glean/preprocess_cat_train_dataset.py --lmdb-path .data/cat --meta-file-path ./data/cat_train/meta_info_LSUNcat_GT.txt --out-dir ./data/cat_train
```

The generated data is stored under `cat_train` and the folder structure is as follows.

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── cat_train
│   │   ├── GT
│   │   ├── BIx8_down
│   │   ├── BIx16_down
│   │   ├── meta_info_LSUNcat_GT.txt
...
```

## Preparing cat_test dataset

1. Download [CAT dataset](https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip) from [here](https://archive.org/details/CAT_DATASET).

2. Download [cat_test/meta_info_Cat100_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/cat_test/meta_info_Cat100_GT.txt) from [GLEAN homepage](https://github.com/ckkelvinchan/GLEAN)

3. Downsample images

Resize the input images to the designated size. We provide such a script:

```shell
python tools/dataset_converters/glean/preprocess_cat_test_dataset.py --data-path ./data/CAT_03 --meta-file-path ./data/cat_test/meta_info_Cat100_GT.txt --out-dir ./data/cat_test
```

The generated data is stored under `cat_test` and the folder structure is as follows.

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── cat_test
│   │   ├── GT
│   │   ├── BIx8_down
│   │   ├── BIx16_down
│   │   ├── meta_info_Cat100_GT.txt
...
```

## Preparing FFHQ dataset

1. Download [FFHQ dataset (images1024x1024)](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) from it's [homepage](https://github.com/NVlabs/ffhq-dataset)

Then you can refactor the folder structure looks like:

```text
ffhq
├── images
|   ├── 00000.png
|   ├── 00001.png
|   ├── ...
|   ├── 69999.png
```

2. Download [ffhq/meta_info_FFHQ_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/FFHQ/meta_info_FFHQ_GT.txt) from [GLEAN homepage](https://github.com/ckkelvinchan/GLEAN)

3. Downsample images

Resize the input images to the designated size. We provide such a script:

```shell
python tools/dataset_converters/glean/preprocess_ffhq_celebahq_dataset.py --data-root ./data/ffhq/images
```

The generated data is stored under `ffhq` and the folder structure is as follows.

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── ffhq
|   |   ├── images
│   │   ├── BIx8_down
|   |   ├── BIx16_down
|   |   ├── meta_info_FFHQ_GT.txt
...
```

## Preparing CelebA-HQ dataset

1. Preparing datasets following it's [homepage](https://github.com/tkarras/progressive_growing_of_gans)

Then you can refactor the folder structure looks like:

```text
CelebA-HQ
├── GT
|   ├── 00000.png
|   ├── 00001.png
|   ├── ...
|   ├── 30000.png
```

2. Download [CelebA-HQ/meta_info_CelebAHQ_val100_GT.txt](https://github.com/ckkelvinchan/GLEAN/blob/main/data/CelebA-HQ/meta_info_CelebAHQ_val100_GT.txt) from [GLEAN homepage](https://github.com/ckkelvinchan/GLEAN)

3. Downsample images

Resize the input images to the designated size. We provide such a script:

```shell
python tools/dataset_converters/glean/preprocess_ffhq_celebahq_dataset.py --data-root ./data/CelebA-HQ/GT
```

The generated data is stored under `CelebA-HQ` and the folder structure is as follows.

```text
mmagic
├── mmagic
├── tools
├── configsdata
├── data
|   ├── CelebA-HQ
|   |   ├── GT
│   │   ├── BIx8_down
|   |   ├── BIx16_down
|   |   ├── meta_info_CelebAHQ_val100_GT.txt
...
```

## Preparing FFHQ_CelebAHQ dataset

We merge FFHQ(`ffhq/images`) and CelebA-HQ(`CelebA-HQ/GT`) to generate FFHQ_CelebAHQ dataset.

The folder structure should looks like:

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── FFHQ_CelebAHQ
|   |   ├── GT
...
```
