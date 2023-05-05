# Preparing NTIRE21 decompression Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{yang2021dataset,
  title={{NTIRE 2021} Challenge on Quality Enhancement of Compressed Video: Dataset and Study},
  author={Ren Yang and Radu Timofte},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}

```

The test datasets can be download from it's [Homepage](https://github.com/RenYang-home/NTIRE21_VEnh).

Please follows the tutorials of the [Homepage](https://github.com/RenYang-home/NTIRE21_VEnh) to generate datasets.

The folder structure should look like:

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── NTIRE21_decompression_track1
|   |   ├── GT
|   |   |   ├── 001
|   |   |   |   ├── 001.png
|   |   |   |   ├── ...
|   |   |   ├── ...
|   |   |   ├── 010
|   |   ├── LQ
|   |   |   ├── 001
|   |   |   |   ├── 001.png
|   |   |   |   ├── ...
|   |   |   ├── ...
|   |   |   ├── 010
|   ├── NTIRE21_decompression_track2
|   |   ├── GT
|   |   ├── LQ
|   ├── NTIRE21_decompression_track3
|   |   ├── GT
|   |   ├── LQ
```
