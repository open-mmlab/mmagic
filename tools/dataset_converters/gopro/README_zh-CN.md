# 准备 GoPro 数据集

<!-- [DATASET] -->

```bibtex
@inproceedings{Zamir2021Restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
  booktitle={CVPR},
  year={2022}
}
```

训练数据集可以从 [此处](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/) 下载。测试数据集可以从 [此处](https://drive.google.com/file/d/1k6DTSHu4saUgrGTYkkZXTptILyG9RRll/) 下载。

文件目录结构应如下所示：

```text
mmagic
├── mmagic
├── tools
├── configs
├── data
|   ├── GoPro
|   |   ├── train
|   |   |   ├── blur
|   |   |   ├── sharp
|   |   ├── test
|   |   |   ├── blur
|   |   |   ├── sharp
```
