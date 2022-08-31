# 图像补全数据集

建议将数据集软链接到 `$MMEDITING/data` 。如果您的文件夹结构不同，您可能需要更改配置文件中的相应路径。

MMEditing 支持的补全数据集：

- [Paris Street View](paris-street-view/README.md) \[ [主页](https://github.com/pathak22/context-encoder/issues/24) \]
- [CelebA-HQ](celeba-hq/README.md) \[ [主页](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) \]
- [Places365](places365/README.md) \[ [主页](http://places2.csail.mit.edu/) \]

由于在图像补全任务中，我们只需要使用图像，因此我们不需要对数据集进行额外的预处理操作，文件目录的结构也可以和本例有所不同。您可以利用原始数据集提供的信息，如 `Place365` （例如 `meta`）。或者，您可以直接遍历数据集文件夹，并将所有图像文件的路径罗列在一个文本文件中。下面的例子节选自 Places365 数据集中的 `Places365_val.txt`，针对图像补全任务，我们只需要使用其中的文件名信息。

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
