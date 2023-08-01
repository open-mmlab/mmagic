# Tutorial 2: Prepare datasets

In this section, we will detail how to prepare data and adopt the proper dataset in our repo for different methods.

We support multiple datasets of different tasks.
There are two ways to use datasets for training and testing models in MMagic:

1. Using downloaded datasets directly
2. Preprocessing downloaded datasets before using them.

The structure of this guide is as follows:

- [Tutorial 2: Prepare datasets](#tutorial-2-prepare-datasets)
  - [Download datasets](#download-datasets)
  - [Prepare datasets](#prepare-datasets)
  - [The overview of the datasets in MMagic](#the-overview-of-the-datasets-in-mmagic)

## Download datasets

You are supposed to download datasets from their homepage first.
Most datasets are available after downloaded, so you only need to make sure the folder structure is correct and further preparation is not necessary.
For example, you can simply prepare Vimeo90K-triplet datasets by downloading datasets from [homepage](http://toflow.csail.mit.edu/).

## Prepare datasets

Some datasets need to be preprocessed before training or testing. We support many scripts to prepare datasets in [tools/dataset_converters](https://github.com/open-mmlab/mmagic/tree/main/tools/dataset_converters). And you can follow the tutorials of every dataset to run scripts.
For example, we recommend cropping the DIV2K images to sub-images. We provide a script to prepare cropped DIV2K dataset. You can run the following command:

```shell
python tools/dataset_converters/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

## The overview of the datasets in MMagic

We support detailed tutorials and split them according to different tasks.

Please check our dataset zoo for data preparation of different tasks.

If you're interested in more details of datasets in MMagic, please check the [advanced guides](../howto/dataset.md).
