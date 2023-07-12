# Preparing SIDD Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{Zamir2021Restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
  booktitle={CVPR},
  year={2022}
}
```

The train datasets can be download from [here](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/). The validation datasets can be download from [here](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/).

For validation datasets, we need to export images from mat file. We provide such a script:

```shell
python tools/dataset_converters/sidd/preprocess_sidd_test_dataset.py --data-root ./data/SIDD/val --out-dir ./data/SIDD/val
```

The folder structure should look like:

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── SIDD
|   |   ├── train
|   |   |   ├── gt
|   |   |   ├── noisy
|   |   ├── val
|   |   |   ├── gt
|   |   |   ├── noisy
|   |   |   ├── ValidationNoisyBlocksSrgb.mat
|   |   |   ├── ValidationGtBlocksSrgb.mat
```
