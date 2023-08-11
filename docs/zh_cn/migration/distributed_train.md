# 分布式训练的迁移

我们已经将[MMGeneration 1.x](https://github.com/open-mmlab/mmgeneration/tree/1.x)合并至MMagic。以下是针对MMGeneration中分布式训练的迁移事项。

在0.x版中，MMGeneration使用`DDPWrapper`和`DynamicRunner`来训练对应的静态和动态模型（例如PGGAN和StyleGANv2），但在1.x 版中，我们使用MMEngine提供的`MMSeparateDistributedDataParallel`来实现分布式训练。

如下是配置前后对比：

<table class="docutils">
    <thead>
      <tr>
        <th> 0.x版中的静态模型 </th>
        <th> 1.x版中的静态模型 </th>
      </tr>
    </thead>
<tbody>
<tr>
<td valign="top">

```python
# 使用DDPWrapper
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False)
```

</td>
<td valign="top">

```python
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)
```

</td>
</tr>
</tbody>
</table>

<table class="docutils">
<thead>
  <tr>
    <th> 0.x版中的动态模型 </th>
    <th> 1.x版中的动态模型 </th>
  </tr>
</thead>
<tbody>
<tr>
<td valign="top">

```python
use_ddp_wrapper = False
find_unused_parameters = False

# 使用DynamicRunner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True)
```

</td>

<td valign="top">

```python
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True) # 针对动态模型，设置`find_unused_parameters`标志为True
```

</td>

</tr>
</tbody>
</table>
