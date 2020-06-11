## Prepare Datasets for Inpainting

It is recommended to symlink the [Places365](https://github.com/CSAILVision/places365) dataset root, the [CelebA-HQ](https://arxiv.org/abs/1710.10196?context=cs.LG) and the [ParisStreedView](https://github.com/pathak22/context-encoder/issues/24) to `$MMEditing/data`:

```
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
│   ├── celeba
│   │   ├── train
|   |   ├── val
│   ├── paris_street_view
│   │   ├── train
|   |   ├── val

```

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
