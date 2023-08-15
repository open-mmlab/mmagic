# Data Structure

MMaigc的数据结构接口`DataSample` 继承自 MMEngine 的 [` BaseDataElement`](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html).MMEngine 的抽象数据接口实现了基础的增/删/改/查功能，且支持不同设备间的数据迁移，也支持了类字典和张量的操作，充分满足了数据的日常使用需求，这也使得不同算法的数据接口可以得到统一。

特别的，`BaseDataElement` 中存在两种类型的数据:

  -  `metainfo` 类型，包含数据的元信息以确保数据的完整性，如 `img_shape`, `img_id` 等数据所在图片的一些基本信息，方便可视化等情况下对数据进行恢复和使用。
  -  `data` 类型，如标注框、框的标签、和实例掩码等。

得益于统一的数据封装，算法库内的 [`visualizer`](https://mmagic.readthedocs.io/zh_CN/latest/user_guides/visualization.html), [`evaluator`](https://mmagic.readthedocs.io/zh_CN/latest/advanced_guides/evaluator.html), [`model`](https://mmagic.readthedocs.io/zh_CN/latest/howto/models.html) 等各个模块间的数据流通都得到了极大的简化。

`DataSample`中的数据分为以下几个属性：

```python
- ``gt_img``: 原始图像
- ``pred_img``: 模型预测图像
- ``ref_img``:参考图像
- ``mask``: 图像修复中的遮挡区域
- ``trimap``: 图像抠图中的三通道图
- ``gt_alpha``: 图像抠图中原始Alpha图
- ``pred_alpha``: 图像抠图中模型预测Alpha图
- ``gt_fg``: 图像抠图中原始前景图
- ``pred_fg``: 图像抠图中模型预测前景图
- ``gt_bg``:  图像抠图中原始背景图
- ``pred_bg``: 图像抠图中模型预测背景图
- ``gt_merged``:  图像抠图中原始合并图
```

以下示例代码展示了 `DataSample` 的组成元素类型：

```python
import torch
import numpy as np
from mmagic.structures import DataSample
img_meta = dict(img_shape=(800, 1196, 3))
img = torch.rand((3, 800, 1196))
data_sample = DataSample(gt_img=img, metainfo=img_meta)
assert 'img_shape' in data_sample.metainfo_keys()
data_sample   
# `DataSample` 的组成元素类型
<DataSample(

        META INFORMATION
        img_shape: (800, 1196, 3)

        DATA FIELDS
        gt_img: tensor(3, 800, 1196)
    ) at 0x1f6a5a99a00>
```

`DataSample`同样支持`stack`和`split`操作对数据进行批处理:



1. Stack

该函数用于将数据样本列表堆叠成一个。当数据样本堆叠时，所有张量字段都将堆叠在第一维度。如果数据样本中有非张量字段，例如列表或字典，则这些字段的值将保存在列表中。

        Args:
            data_samples (Sequence['DataSample']): 待堆叠的数据样本序列
    
        Returns:
            DataSample: 堆叠的数据样本

2. Split

该函数将在第一维度拆分数据样本序列。

```
	Args:
         allow_nonseq_value (bool): 是否允许在拆分操作中使用非顺序数据。如果为 "True"，			将为所有拆分数据样本复制非序列数据；否则，将引发错误。默认为 "False"。

    Returns:
         Sequence[DataSample]: 拆分后的数据样本列表。
```

以下示例代码展示了 `stack`和`split` 的使用方法：

```py
import torch
import numpy as np
from mmagic.structures import DataSample
img_meta1 = img_meta2 = dict(img_shape=(800, 1196, 3))
img1 = torch.rand((3, 800, 1196))
img2 = torch.rand((3, 800, 1196))
data_sample1 = DataSample(gt_img=img1, metainfo=img_meta1)
data_sample2 = DataSample(gt_img=img2, metainfo=img_meta1)
```

```py
# stack them and then use as batched-tensor!
data_sample = DataSample.stack([data_sample1, data_sample2])
print(data_sample.gt_img.shape)
    torch.Size([2, 3, 800, 1196])
print(data_sample.metainfo)
    {'img_shape': [(800, 1196, 3), (800, 1196, 3)]}

# split them if you want
data_sample1_, data_sample2_ = data_sample.split()
assert (data_sample1_.gt_img == img1).all()
assert (data_sample2_.gt_img == img2).all()
```



---

## English

`DataSample` , the data structure interface of MMagic, inherits from [` BaseDataElement`](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html). The base class has implemented basic add/delete/update/check functions and supports data migration between different devices, as well as dictionary-like and tensor-like operations, which also allows the interfaces of different algorithms to be unified.

 Specifically, an instance of BaseDataElement consists of two components: 

  - ``metainfo``, which contains some meta information,
    e.g., `img_shape`, `img_id`, `color_order`, etc.
  - ``data``, which contains the data used in the loop.

Thanks to ` DataSample` , the data flow between each module in the algorithm libraries, such as [`visualizer`](https://mmagic.readthedocs.io/en/latest/user_guides/visualization.html), [`evaluator`](https://mmagic.readthedocs.io/en/latest/advanced_guides/evaluator.html), [`model`](https://mmagic.readthedocs.io/en/latest/howto/models.html), is greatly simplified.



The attributes in ``DataSample`` are divided into several parts:

```python
- ``gt_img``: Ground truth image(s).
- ``pred_img``: Image(s) of model predictions.
- ``ref_img``: Reference image(s).
- ``mask``: Mask in Inpainting.
- ``trimap``: Trimap in Matting.
- ``gt_alpha``: Ground truth alpha image in Matting.
- ``pred_alpha``: Predicted alpha image in Matting.
- ``gt_fg``: Ground truth foreground image in Matting.
- ``pred_fg``: Predicted foreground image in Matting.
- ``gt_bg``: Ground truth background image in Matting.
- ``pred_bg``: Predicted background image in Matting.
- ``gt_merged``: Ground truth merged image in Matting.
```

The following sample code demonstrates the components of `DataSample`:


```python
     >>> import torch
     >>> import numpy as np
     >>> from mmagic.structures import DataSample
     >>> img_meta = dict(img_shape=(800, 1196, 3))
     >>> img = torch.rand((3, 800, 1196))
     >>> data_sample = DataSample(gt_img=img, metainfo=img_meta)
     >>> assert 'img_shape' in data_sample.metainfo_keys()
     >>> data_sample
	 >>># metainfo and data of DataSample         
    <DataSample(

        META INFORMATION
        img_shape: (800, 1196, 3)

        DATA FIELDS
        gt_img: tensor(3, 800, 1196)
    ) at 0x1f6a5a99a00>
```

  We also support `stack` and `split` operation to handle a batch of data samples.

1. Stack

Stack a list of data samples to one. All tensor fields will be stacked at first dimension. Otherwise the values will be saved in a list.

        Args:
            data_samples (Sequence['DataSample']): A sequence of `DataSample` to stack.
    
        Returns:
            DataSample: The stacked data sample.

2. Split

Split a sequence of data sample in the first dimension.

```
	Args:
         allow_nonseq_value (bool): Whether allow non-sequential data in
         split operation. If True, non-sequential data will be copied
         for all split data samples. Otherwise, an error will be
         raised. Defaults to False.

    Returns:
         Sequence[DataSample]: The list of data samples after splitting.
```

The following sample code demonstrates the use of `stack` and ` split`:

```py
import torch
import numpy as np
from mmagic.structures import DataSample
img_meta1 = img_meta2 = dict(img_shape=(800, 1196, 3))
img1 = torch.rand((3, 800, 1196))
img2 = torch.rand((3, 800, 1196))
data_sample1 = DataSample(gt_img=img1, metainfo=img_meta1)
data_sample2 = DataSample(gt_img=img2, metainfo=img_meta1)
```

```py
# stack them and then use as batched-tensor!
data_sample = DataSample.stack([data_sample1, data_sample2])
print(data_sample.gt_img.shape)
    torch.Size([2, 3, 800, 1196])
print(data_sample.metainfo)
    {'img_shape': [(800, 1196, 3), (800, 1196, 3)]}

# split them if you want
data_sample1_, data_sample2_ = data_sample.split()
assert (data_sample1_.gt_img == img1).all()
assert (data_sample2_.gt_img == img2).all()
```