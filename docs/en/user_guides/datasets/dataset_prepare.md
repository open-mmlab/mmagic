# Prepare Datasets

We supports multiple datasets of different tasks.
There are two ways to use datasets for training and testing models in MMEditing:

1. Using downloaded datasets directly
2. Preprocessing downloaded datasets before using them.

## Download datasets

You are supposed to download datasets from their homepage first.
Most of datasets are available after downloaded, so you only need to make sure the folder structure is correct and further preparation is not necessary.
For example, you can simply prepare [Vimeo90K-triplet](./video_interpolation_datasets.md#Vimeo90K-triplet-Dataset) datasets by downloading datasets from [homepage](http://toflow.csail.mit.edu/).

## Prepare datasets

Some datasets need to be preprocessed before training or testing. We support many scripts to prepare datasets in [tools/data](/tools/data). And you can follow the tutorials of every dataset to run scripts.
For example, we recommend to crop the DIV2K images to sub-images. We provide a script to prepare cropped DIV2K dataset. You can run following command:

```shell
python tools/data/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

## The overview of the datasets in MMEditing

We support detailed tutorials and split them according to different tasks. Please follow the corresponding tutorials for data preparation of different tasks.

- [Prepare Inpainting Datasets](inpainting_datasets.md)
  - [Paris Street View](inpainting_datasets.md#paris-street-view-dataset) \[ [Homepage](https://github.com/pathak22/context-encoder/issues/24) \]
  - [CelebA-HQ](inpainting_datasets.md#celeba-hq-dataset) \[ [Homepage](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) \]
  - [Places365](inpainting_datasets.md#places365-dataset) \[ [Homepage](http://places2.csail.mit.edu/) \]
- [Prepare Matting Datasets](matting_datasets.md)
  - [Composition-1k](matting_datasets.md#composition-1k-dataset) \[ [Homepage](https://sites.google.com/view/deepimagematting) \]
- [Prepare Super-Resolution Datasets](super_resolution_datasets.md)
  - [DF2K_OST](super_resolution_datasets.md#df2kost-dataset) \[ [Homepage](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md) \]
  - [DIV2K](super_resolution_datasets.md#div2k-dataset) \[ [Homepage](https://data.vision.ee.ethz.ch/cvl/DIV2K/) \]
  - [REDS](super_resolution_datasets.md#reds-dataset) \[ [Homepage](https://seungjunnah.github.io/Datasets/reds.html) \]
  - [Vimeo90K](super_resolution_datasets.md#vimeo90k-dataset) \[ [Homepage](http://toflow.csail.mit.edu) \]
- [Prepare Video Frame Interpolation Datasets](video_interpolation_datasets.md)
  - [Vimeo90K-triplet](video_interpolation_datasets.md#vimeo90k-triplet-dataset) \[ [Homepage](http://toflow.csail.mit.edu) \]
