# Tutorial 2: Prepare Datasets

In this section, we will detail how to prepare data and adopt proper dataset in our repo for different methods.

We supports multiple datasets of different tasks.
There are two ways to use datasets for training and testing models in MMEditing:

1. Using downloaded datasets directly
2. Preprocessing downloaded datasets before using them.

The structure of this guide are as follows:

- [Download datasets](#download-datasets)
- [Prepare datasets](#prepare-datasets)
- [The overview of the datasets in MMEditing](#the-overview-of-the-datasets-in-mmediting)

## Download datasets

You are supposed to download datasets from their homepage first.
Most of datasets are available after downloaded, so you only need to make sure the folder structure is correct and further preparation is not necessary.
For example, you can simply prepare [Vimeo90K-triplet](./video_interpolation_datasets.md#Vimeo90K-triplet-Dataset) datasets by downloading datasets from [homepage](http://toflow.csail.mit.edu/).

## Prepare datasets

Some datasets need to be preprocessed before training or testing. We support many scripts to prepare datasets in [tools/dataset_converters](/tools/dataset_converters). And you can follow the tutorials of every dataset to run scripts.
For example, we recommend to crop the DIV2K images to sub-images. We provide a script to prepare cropped DIV2K dataset. You can run following command:

```shell
python tools/dataset_converters/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

## The overview of the datasets in MMEditing

We support detailed tutorials and split them according to different tasks. Please follow the corresponding tutorials for data preparation of different tasks.

- [Prepare Inpainting Datasets](./datasets/inpainting_datasets.md)
  - [Paris Street View](./datasets/inpainting_datasets.md#paris-street-view-dataset) \[ [Homepage](https://github.com/pathak22/context-encoder/issues/24) \]
  - [CelebA-HQ](./datasets/inpainting_datasets.md#celeba-hq-dataset) \[ [Homepage](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) \]
  - [Places365](./datasets/inpainting_datasets.md#places365-dataset) \[ [Homepage](http://places2.csail.mit.edu/) \]
- [Prepare Matting Datasets](./datasets/matting_datasets.md)
  - [Composition-1k](./datasets/matting_datasets.md#composition-1k-dataset) \[ [Homepage](https://sites.google.com/view/deepimagematting) \]
- [Prepare Super-Resolution Datasets](./datasets/super_resolution_datasets.md)
  - [DF2K_OST](./datasets/super_resolution_datasets.md#df2kost-dataset) \[ [Homepage](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md) \]
  - [DIV2K](./datasets/super_resolution_datasets.md#div2k-dataset) \[ [Homepage](https://data.vision.ee.ethz.ch/cvl/DIV2K/) \]
  - [REDS](./datasets/super_resolution_datasets.md#reds-dataset) \[ [Homepage](https://seungjunnah.github.io/Datasets/reds.html) \]
  - [Vimeo90K](./datasets/super_resolution_datasets.md#vimeo90k-dataset) \[ [Homepage](http://toflow.csail.mit.edu) \]
- [Prepare Video Frame Interpolation Datasets](./datasets/video_interpolation_datasets.md)
  - [Vimeo90K-triplet](./datasets/video_interpolation_datasets.md#vimeo90k-triplet-dataset) \[ [Homepage](http://toflow.csail.mit.edu) \]
- [Prepare Unconditional GANs Datasets](./datasets/unconditional_gans_datasets.md)
- [Prepare Image Translation Datasets](./datasets/image_translation_datasets.md)
