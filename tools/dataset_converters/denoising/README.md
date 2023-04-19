# Preparing Denoising Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{Zamir2021Restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
  booktitle={CVPR},
  year={2022}
}
```

The test datasets (Set12, BSD68, CBSD68, Kodak, McMaster, Urban100) can be download from [here](https://drive.google.com/file/d/1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0/).

The folder structure should look like:

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── denoising_gaussian_test
|   |   ├── Set12
|   |   ├── BSD68
|   |   ├── CBSD68
|   |   ├── Kodak
|   |   ├── McMaster
|   |   ├── Urban100
```
