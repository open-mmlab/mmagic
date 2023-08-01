# 常见问题解答

我们在此列出了许多用户面临的一些常见问题及其相应的解决方案。如果您发现任何常见问题，并有办法帮助他人解决这些问题，请随时丰富列表内容。如果这里的内容没有涵盖您的问题，请使用[提供的模板](https://github.com/open-mmlab/mmagic/issues/new/choose)创建一个问题，并确保填写了模板中的所有必要信息。

## 常见问题

**问题1：** “xxx: ‘yyy is not in the zzz registry’”.

**回答1：** 只有导入模块文件时，才会触发注册表机制。所以你需要在某个地方导入该文件。

**问题2：** 某个数据集的文件夹结构是什么？

**回答2：** 您可以根据[数据集准备](https://github.com/sijiua/mmagic/blob/dev-1.x/docs/en/user_guides/dataset_prepare.md)教程来确保文件夹结构的正确性。

**问题3：** 如何使用 LMDB 数据训练模型？

**回答3：** 您可以使用工具/数据中的脚本制作 LMDB 文件。更多详情请参见[数据集准备](https://github.com/sijiua/mmagic/blob/dev-1.x/docs/en/user_guides/dataset_prepare.md)教程。

**问题4：** 为什么使用了 MMCV==xxx，但在导入 mmagic 时却出现了不兼容？

**回答4：** 这是因为 MMCV 和 MMagic 的版本不兼容。兼容的 MMagic 和 MMCV 版本如下所示。请选择正确的 MMCV 版本以避免安装问题。

| MMagic版本 |    MMCV 版本     |
| :--------: | :--------------: |
|   master   | mmcv-full>=2.0.0 |

注意：如果已安装 mmcv，则需要先运行 pip uninstall mmcv。如果同时安装了 mmcv 和 mmcv-full，则会出现模块未找到错误（ModuleNotFoundError）。
**问题5：** 如何忽略基本配置中的某些字段？

**回答5：** 有些时候您可以设置 _delete_=True 来忽略基本配置中的某些字段。您可以参考 [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md#delete-key-in-dict) 的简单说明。
您可以仔细阅读[本教程](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/config.md)，以便更好地理解这一功能。

**问题6：** 如何在配置中使用中间变量？

**回答6：** 有些中间变量会在配置文件中使用，比如数据集中的 train_pipeline/test_pipeline。值得注意的是，当修改子配置中的中间变量时，用户需要再次将中间变量传递到相应的字段中。
