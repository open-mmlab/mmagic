# 补全数据集

建议将数据集软链接到 `$MMEDITING/data` 。如果您的文件夹结构不同，您可能需要更改配置文件中的相应路径。

MMEditing 支持的补全数据集：

* [Paris Street View](paris-street-view/README.md) \[ [主页](https://github.com/pathak22/context-encoder/issues/24) \]
* [CelebA-HQ](celeba-hq/README.md) \[ [主页](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) \]
* [Places365](places365/README.md) \[ [主页](http://places2.csail.mit.edu/) \]

由于我们只需要用于补全任务的图像，因此不需要进一步准备，而且文件夹结构亦可以与示例不同。您可以利用原始数据集提供的信息，如 `Place365` （例如 `meta` ）。此外，您亦可以轻松扫描数据集并将所有图像列出到特定的 `txt` 文件中。 以下是 Places365 中的 `Places365_val.txt` 的示例，我们将仅在补全中使用图像名称信息。

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
