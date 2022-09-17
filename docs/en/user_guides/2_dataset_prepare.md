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
For example, you can simply prepare [Vimeo90K-triplet](../dataset_zoo/4_video_interpolation_datasets.md#vimeo90K-triplet-dataset) datasets by downloading datasets from [homepage](http://toflow.csail.mit.edu/).

## Prepare datasets

Some datasets need to be preprocessed before training or testing. We support many scripts to prepare datasets in [tools/dataset_converters](/tools/dataset_converters). And you can follow the tutorials of every dataset to run scripts.
For example, we recommend to crop the DIV2K images to sub-images. We provide a script to prepare cropped DIV2K dataset. You can run following command:

```shell
python tools/dataset_converters/super-resolution/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

## The overview of the datasets in MMEditing

We support detailed tutorials and split them according to different tasks.

Please check our [the corresponding tutorials](../dataset_zoo/0_overview.md) for data preparation of different tasks.
