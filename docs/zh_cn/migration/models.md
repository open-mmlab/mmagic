# 模型的迁移

我们在 MMagic 1.x. 版本更新了模型设定，其中重要的改动如下所示：

- 删除 `pretrained` 字段.
- 在模型设定中添加 `train_cfg` 和 `test_cfg` 字段.
- 添加 `data_preprocessor` 字段. 这里主要是将归一化和颜色空间转换操作从 `dataset transform` 流程中移动到 `data_preprocessor` 中. 我们接下来会介绍`data_preprocessor`.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
model = dict(
    type='BasicRestorer',  # Name of the model
    generator=dict(  # Config of the generator
        type='EDSR',  # Type of the generator
        in_channels=3,  # Channel number of inputs
        out_channels=3,  # Channel number of outputs
        mid_channels=64,  # Channel number of intermediate features
        num_blocks=16,  # Block number in the trunk network
        upscale_factor=scale, # Upsampling factor
        res_scale=1,  # Used to scale the residual in residual block
        rgb_mean=(0.4488, 0.4371, 0.4040),  # Image mean in RGB orders
        rgb_std=(1.0, 1.0, 1.0)),  # Image std in RGB orders
    pretrained=None,
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))  # Config for pixel loss model training and testing settings
```

</td>

<td valign="top">

```python
model = dict(
    type='BaseEditModel',  # Name of the model
    generator=dict(  # Config of the generator
        type='EDSRNet',  # Type of the generator
        in_channels=3,  # Channel number of inputs
        out_channels=3,  # Channel number of outputs
        mid_channels=64,  # Channel number of intermediate features
        num_blocks=16,  # Block number in the trunk network
        upscale_factor=scale, # Upsampling factor
        res_scale=1,  # Used to scale the residual in residual block
        rgb_mean=(0.4488, 0.4371, 0.4040),  # Image mean in RGB orders
        rgb_std=(1.0, 1.0, 1.0)),  # Image std in RGB orders
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean')  # Config for pixel loss
    train_cfg=dict(),  # Config of training model.
    test_cfg=dict(),  # Config of testing model.
    data_preprocessor=dict(  # The Config to build data preprocessor
        type='DataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))
```

</td>

</tr>
</thead>
</table>

我们在 MMagic 1.x. 版本中对模型进行了重构，其中重要的改动如下所示：

- MMagic 1.x 中的 `models` 被重构为六个部分：`archs`、`base_models`、`data_preprocessors`、`editors`、`diffusion_schedulers` 和 `losses`.

- 在 `models` 中添加了 `data_preprocessor` 模块。这里主要是将归一化和颜色空间转换操作从 `dataset transform` 流程中移动到 `data_preprocessor` 中.此时，数据流经过数据预处理后，会先经过 `data_preprocessor` 模块的转换，然后再输入到模型中.

模型的更多详细信息请参见[模型指南](../howto/models.md).
