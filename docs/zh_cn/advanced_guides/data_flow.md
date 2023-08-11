# 数据流

- [数据流](#数据流)
  - [数据流概述](#数据流概述)
  - [数据集与模型之间的数据流](#数据集与模型之间的数据流)
    - [数据加载器的数据处理](#数据加载器的数据处理)
    - [数据预处理器的数据处理](#数据预处理器的数据处理)
  - [模型输出与可视化器之间的数据流](#模型输出与可视化器之间的数据流)

## 数据流概述

[Runner](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/runner.md) 相当于 MMEngine 中的“集成器”。它覆盖了框架的所有方面，并肩负着组织和调度几乎所有模块的责任，这意味着各模块之间的数据流也由 `Runner` 控制。在本章节中，我们将介绍 [Runner](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html) 管理的内部模块之间的数据流和数据格式约定。

<div align="center">
<img src="https://github.com/open-mmlab/mmagic/assets/36404164/fc6ab53c-8804-416d-94cd-332c533a07ad" height="150" />
</div>

在上图中，在训练迭代中，数据加载器（dataloader）从存储中加载图像并传输到数据预处理器（data preprocessor），数据预处理器会将图像放到特定的设备上，并将数据堆叠到批处理中，之后模型接受批处理数据作为输入，最后将模型的输出计算损失函数（loss）。在评估时模型参数会被冻结，模型的输出需要经由数据预处理器（data preprocessor）解构再被传递给 [Evaluator](./evaluation.md#ioumetric)计算指标或者提供给[Visualizer](../user_guides/visualization.md)进行可视化。

## 数据集与模型之间的数据流

在本节中将介绍在MMagic中数据集中的数据流传递，关于[数据集定义](https://mmagic.readthedocs.io/zh_CN/latest/howto/dataset.html)和[数据处理管线](https://mmagic.readthedocs.io/zh_CN/latest/howto/transforms.html)相关的解读详见开发指南。数据集 （dataset） 和模型 （model）之间的数据流传递一般可以分为如下四个步骤 :

1. 读取 `XXDataset` 收集数据集的原始信息，并且通过数据处理管线对数据进行数据转换处理;

2. 使用 `PackInputs` 将转换完成的数据打包成为一个字典;

3. 使用 `collate_fn` 将各个张量集成为一个批处理张量;

4. 使用 `data_preprocessor` 把以上所有数据迁移到 GPUS 等目标设备，并在数据加载器中将之前打包的字典解压为一个元组，该元组包含输入图像与对应的元信息（`DataSample`）。

### 数据处理管线的数据处理

在MMagic中，经由不同类型的`XXDataset`, 分别读取数据(LQ)以及标注(GT)，并且在不同的数据预处理管道中进行数据转换，最后通过`PackInputs`将处理之后的数据打包为字典，此字典包含训练以及测试过程所需的所有数据。

<table class="docutils">
<thead>
  <tr>
    <th> base_edit_model.py </th>
    <th> base_conditional_gan.py </th>
<tbody>
<tr>
<td valign="top">

```python
@MODELS.register_module()
class BaseEditModel(BaseModel):
    """Base model for image and video editing.
    """
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs) -> Union[torch.Tensor, List[DataSample], dict]:
        if isinstance(inputs, dict):
            inputs = inputs['img']
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'predict':
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)
```

</td>

<td valign="top">

```python
@MODELS.register_module()
class BaseConditionalGAN(BaseGAN):
    """Base class for Conditional GAM models.
    """
    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[DataSample]:
        if isinstance(inputs, Tensor):
            noise = inputs
            sample_kwargs = {}
        else:
            noise = inputs.get('noise', None)
            num_batches = get_valid_num_batches(inputs, data_samples)
            noise = self.noise_fn(noise, num_batches=num_batches)
            sample_kwargs = inputs.get('sample_kwargs', dict())
        num_batches = noise.shape[0]

        pass
        ...
```

</td>

</tr>
</thead>
</table>

例如在`BaseEditModel`和`BaseConditionalGAN`模型中分别需要输入（input）包括 `img` 和 `noise` 的键值输入。同时，相应的字段也应该在配置文件中暴露, 以[cyclegan_lsgan-id0-resnet-in_1xb1-80kiters_facades.py](../../../configs/cyclegan/cyclegan_lsgan-id0-resnet-in_1xb1-80kiters_facades.py)为例，

```python
domain_a = 'photo'
domain_b = 'mask'
pack_input = dict(
    type='PackInputs',
    keys=[f'img_{domain_a}', f'img_{domain_b}'],
    data_keys=[f'img_{domain_a}', f'img_{domain_b}'])
```

### 数据加载器的数据处理

以数据集中的获取字典列表作为输入，数据加载器（dataloader）中的 `collect_fn` 会提取每个字典的`inputs`并将其整合成一个批处理张量；此外，每个字典中的`data_sample`也会被整合为一个列表，从而输出一个与先前字典有相同键的字典；最终数据加载器会通过 `collect_fn` 输出这个字典。详细文档可见[数据集与数据加载器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/dataset.html)。

### 数据预处理器的数据处理

数据预处理是数据输入模型之前，处理数据过程的最后一步。 数据预处理过程会对图像进行归一处理，如把 BGR 模式转换为 RGB 模式，并将所有数据迁移至 GPU 等目标设备中 。上述各步骤完成后，最终会得到一个元组，该元组包含一个批处理图像的列表，和一个数据样本的列表。详细文档可见[数据预处理](./data_preprocessor.md)。

## 模型输出与可视化器之间的数据流

MMEngine约定了[抽象数据接口](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/data_element.md)用于数据传递，其中 [数据样本](./structures.md)(DataSample) 作为一层更加高级封装可以容纳更多类别的标签数据。在MMagic中，用于可视化对比的`ConcatImageVisualizer`同时也通过 `add_datasample` 方法控制可视化具体内容，具体配置如下。

```python
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
```
