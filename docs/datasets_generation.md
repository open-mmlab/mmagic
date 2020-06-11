## Prepare Datasets for Generation

It is recommended to symlink the dataset root of the paired dataset to `$MMEditing/data/paired`, and the dataset root of the unpaired dataset to `$MMEditing/data/unpaired`:

```
mmediting
├── mmedit
├── tools
├── configs
├── data
│   ├── paired
│   │   ├── facades
│   │   ├── maps
|   |   ├── edges2shoes
|   |   |    ├── train
|   |   |    ├── test
│   ├── unpaired
│   │   ├── facades
|   |   ├── horse2zebra
|   |   ├── summer2winter_yosemite
|   |   |    ├── trainA
|   |   |    ├── trainB
|   |   |    ├── testA
|   |   |    ├── testB
```

Currently, you can directly download paired datasets from [here](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/), and unpaired datasets from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.

As we only need images and the correct folder structure for generation task, further preparation is not necessary. For paired datasets, each sample should be paired images that are concatenated in the width dimension. For unpaired datasets, `trainA` and `testA` contain images from domain A, whereas `trainB` and `testB` contain images from domain B. We recommend you to download the well-prepared datasets directly and conduct experiments. Or you can just put your images in the right place.
