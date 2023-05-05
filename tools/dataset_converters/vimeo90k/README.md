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

The training and test datasets can be downloaded from [here](http://toflow.csail.mit.edu/).

Then you can rename the directory `vimeo_septuplet/sequences` to  `vimeo90k/GT`. The Vimeo90K dataset has a `clip/sequence/img` folder structure:

```text
vimeo90k
├── GT
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
├── sep_trainlist.txt
├── sep_testlist.txt
```

To generate the downsampling images BIx4 and BDx4 and prepare the annotation file, you need to run the following command:

```shell
python tools/dataset_converters/vimeo90k/preprocess_vimeo90k_dataset.py --data-root ./data/vimeo90k
```

The folder structure should look like:

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
│   ├── vimeo_triplet
│   │   ├── GT
│   │   │   ├── 00001
│   │   │   │   ├── 0001
│   │   │   │   │   ├── im1.png
│   │   │   │   │   ├── im2.png
│   │   │   │   │   ├── ...
│   │   │   │   ├── 0002
│   │   │   │   ├── 0003
│   │   │   │   ├── ...
│   │   │   ├── 00002
│   │   │   ├── ...
│   │   ├── BIx4
│   │   ├── BDx4
│   │   ├── meta_info_Vimeo90K_test_GT.txt
│   │   ├── meta_info_Vimeo90K_train_GT.txt
```

## Prepare LMDB dataset for Vimeo90K

If you want to use LMDB datasets for faster IO speed, you can make LMDB files by:

```shell
python tools/dataset_converters/vimeo90k/preprocess_vimeo90k_dataset.py --data-root ./data/vimeo90k --train_list ./data/vimeo90k/sep_trainlist.txt --gt-path ./data/vimeo90k/GT --lq-path ./data/Vimeo90k/BIx4  --make-lmdb
```
