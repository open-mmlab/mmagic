# Data Structure

`DataSample` , the data structure interface of MMagic, inherits from [` BaseDataElement`](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html). The base class has implemented basic add/delete/update/check functions and supports data migration between different devices, as well as dictionary-like and tensor-like operations, which also allows the interfaces of different algorithms to be unified.

Specifically, an instance of BaseDataElement consists of two components:

- `metainfo`, which contains some meta information,
  e.g., `img_shape`, `img_id`, `color_order`, etc.
- `data`, which contains the data used in the loop.

Thanks to ` DataSample` , the data flow between each module in the algorithm libraries, such as [`visualizer`](https://mmagic.readthedocs.io/en/latest/user_guides/visualization.html), [`evaluator`](https://mmagic.readthedocs.io/en/latest/advanced_guides/evaluator.html), [`model`](https://mmagic.readthedocs.io/en/latest/howto/models.html), is greatly simplified.

The attributes in `DataSample` are divided into several parts:

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

```
    Args:
        data_samples (Sequence['DataSample']): A sequence of `DataSample` to stack.

    Returns:
        DataSample: The stacked data sample.
```

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
