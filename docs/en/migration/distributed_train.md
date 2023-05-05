# Migration of Distributed Training Settings

We have merged [MMGeneration 1.x](https://github.com/open-mmlab/mmgeneration/tree/1.x) into MMagic. Here is migration of Distributed Training Settings about MMGeneration.

In 0.x version, MMGeneration uses `DDPWrapper` and `DynamicRunner` to train static and dynamic model (e.g., PGGAN and StyleGANv2) respectively. In 1.x version, we use `MMSeparateDistributedDataParallel` provided by MMEngine to implement distributed training.

The configuration differences are shown below:

<table class="docutils">
<thead>
  <tr>
    <th> Static Model in 0.x Version </th>
    <th> Static Model in 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
# Use DDPWrapper
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
</thead>
</table>

<table class="docutils">
<thead>
  <tr>
    <th> Dynamic Model in 0.x Version </th>
    <th> Dynamic Model in 1.x Version </th>
<tbody>
<tr>

<td valign="top">

```python
use_ddp_wrapper = False
find_unused_parameters = False

# Use DynamicRunner
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
    find_unused_parameters=True) # set `find_unused_parameters` for dynamic models
```

</td>

</tr>
</thead>
</table>
