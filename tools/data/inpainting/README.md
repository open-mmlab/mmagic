# Inpainting Datasets

It is recommended to symlink the dataset root to `$MMEDITING/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

MMEditing supported inpainting datasets:

- [Paris Street View](paris-street-view/README.md) \[ [Homepage](https://github.com/pathak22/context-encoder/issues/24) \]
- [CelebA-HQ](celeba-hq/README.md) \[ [Homepage](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) \]
- [Places365](places365/README.md) \[ [Homepage](http://places2.csail.mit.edu/) \]

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
