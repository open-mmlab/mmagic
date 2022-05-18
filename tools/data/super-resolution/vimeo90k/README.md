# Preparing Vimeo90K Dataset

<!-- [DATASET] -->

```bibtex
@article{xue2019video,
  title={Video Enhancement with Task-Oriented Flow},
  author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  journal={International Journal of Computer Vision (IJCV)},
  volume={127},
  number={8},
  pages={1106--1125},
  year={2019},
  publisher={Springer}
}
```

The training and test datasets can be download from [here](http://toflow.csail.mit.edu/).

The Vimeo90K dataset has a `clip/sequence/img` folder structure:

```text
├── GT/LQ
│   ├── 00001
│   │   ├── 0001
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   ├── ...
│   │   ├── 0002
│   │   ├── 0003
│   │   ├── ...
│   ├── 00002
│   ├── ...
```

## Prepare the annotation files for Vimeo90K dataset

To prepare the annotation file for training, you need to download the official training list path for Vimeo90K from the official website, and run the following command:

```shell
python tools/data/super-resolution/vimeo90k/preprocess_vimeo90k_dataset.py ./data/Vimeo90K/official_train_list.txt
```

The annotation file for test is generated similarly.

## Prepare LMDB dataset for Vimeo90K

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/data/super-resolution/vimeo90k/preprocess_vimeo90k_dataset.py ./data/Vimeo90K/official_train_list.txt --gt-path ./data/Vimeo90K/GT --lq-path ./data/Vimeo90K/LQ  --make-lmdb
```
