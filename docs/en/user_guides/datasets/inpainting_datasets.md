# Inpainting Datasets

It is recommended to symlink the dataset root to `$MMEDITING/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

MMEditing supported inpainting datasets:

- [Paris Street View](#paris-street-view-dataset) \[ [Homepage](https://github.com/pathak22/context-encoder/issues/24) \]
- [CelebA-HQ](#celeba-hq-dataset) \[ [Homepage](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) \]
- [Places365](#places365-dataset) \[ [Homepage](http://places2.csail.mit.edu/) \]

As we only need images for inpainting task, further preparation is not necessary and the folder structure can be different from the example. You can utilize the information provided by the original dataset like `Place365` (e.g. `meta`). Also, you can easily scan the data set and list all of the images to a specific `txt` file. Here is an example for the `Places365_val.txt` from Places365 and we will only use the image name information in inpainting.

```
Places365_val_00000001.jpg 165
Places365_val_00000002.jpg 358
Places365_val_00000003.jpg 93
Places365_val_00000004.jpg 164
Places365_val_00000005.jpg 289
Places365_val_00000006.jpg 106
Places365_val_00000007.jpg 81
Places365_val_00000008.jpg 121
Places365_val_00000009.jpg 150
Places365_val_00000010.jpg 302
Places365_val_00000011.jpg 42
```

## CelebA-HQ Dataset

<!-- [DATASET] -->

```bibtex
@article{karras2017progressive,
  title={Progressive growing of gans for improved quality, stability, and variation},
  author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  journal={arXiv preprint arXiv:1710.10196},
  year={2017}
}
```

Follow the instructions [here](https://github.com/tkarras/progressive_growing_of_gans##preparing-datasets-for-training) to prepare the dataset.

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── celeba-hq
│   │   ├── train
|   |   ├── val

```

## Paris Street View Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{pathak2016context,
  title={Context encoders: Feature learning by inpainting},
  author={Pathak, Deepak and Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2536--2544},
  year={2016}
}
```

Obtain the dataset [here](https://github.com/pathak22/context-encoder/issues/24).

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── paris_street_view
│   │   ├── train
|   |   ├── val

```

## Places365 Dataset

<!-- [DATASET] -->

```bibtex
 @article{zhou2017places,
   title={Places: A 10 million Image Database for Scene Recognition},
   author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   year={2017},
   publisher={IEEE}
 }

```

Prepare the data from [Places365](http://places2.csail.mit.edu/download.html).

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── places
│   │   ├── test_set
│   │   ├── train_set
|   |   ├── meta
|   |   |    ├── Places365_train.txt
|   |   |    ├── Places365_val.txt
```
