# 调度器的迁移

我们更新了MMagic 1.x 中的调度器设置，重要修改如下:

- 现在我们使用 `optim_wrapper` 字段来指定关于优化过程的所有配置。`optimizer` 字段现在是 `optim_wrapper` 的一个子字段。
- `lr_config` 字段被移除，我们使用新的 `param_scheduler` 来代替它。
- `total_iters` 字段已移至 `train_cfg`，作为 `max_iters`, `val_cfg` 和 `test_cfg`，用于配置训练、验证和测试中的循环。

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))   # 用于构建优化器的配置，支持 PyTorch 中的所有优化器，其参数与 PyTorch 中的参数相同。
total_iters = 300000 # 总训练迭代次数
lr_config = dict( # 用于注册 LrUpdater hook 的学习率调度器配置
    policy='Step', by_epoch=False, step=[200000], gamma=0.5)  # 调度器的策略
```

</td>

<td valign="top">

```python
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4),
    )
)  # 用于构建优化器的配置，支持 PyTorch 中的所有优化器，其参数与 PyTorch 中的参数相同。
param_scheduler = dict(  # 学习策略的配置
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)  # 调度器的策略
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # 训练循环类型的配置
val_cfg = dict(type='ValLoop')  # 验证循环类型的名称
test_cfg = dict(type='TestLoop')  # 测试循环类型的名称
```

</td>

</tr>
</thead>
</table>

> 有关调度器设置的更多详细信息可在 [MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/en/migration/param_scheduler.md) 中找到。
