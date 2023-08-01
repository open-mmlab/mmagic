# 评估与测试设置的迁移

我们更新了 MMagic 1.x 中的评估设置，重要修改如下：

- 评估字段被分为 `val_evaluator` 和 `test_evaluator` ， `interval` 被移动到 `train_cfg.val_interval` 。
- 评估指标从 `test_cfg` 移至 `val_evaluator` 和 `test_evaluator`

<table class="docutils">
<thead>
  <tr>
    <th> 原评估配置 </th>
    <th> 新评估配置 </th>
<tbody>
<tr>
<td valign="top">

```python
train_cfg = None  # 训练配置字典变量设为 None
test_cfg = dict(  # 测试配置字典变量
    metrics=['PSNR'],  # 测试期间使用的指标 PSNR （峰值信噪比）
    crop_border=scale)  # 评估期间裁剪边框

evaluation = dict(  # 构建评估钩子的配置字典变量
    interval=5000,  # 评价间隔
    save_image=True,  # 评估期间保存图像
    gpu_collect=True)  # 使用 GPU 收集
```

</td>

<td valign="top">

```python
val_evaluator = [
    dict(type='PSNR', crop_border=scale),  # 要评估的指标名称
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # 训练循环类型配置
val_cfg = dict(type='ValLoop')  # 验证循环类型的名称
test_cfg = dict(type='TestLoop')  # 测试循环类型的名称
```

</td>

</tr>
</thead>
</table>

我们已将[MMGeneration 1.x](https://github.com/open-mmlab/mmgeneration/tree/1.x)合并到 MMagic 中.
这里是关于 MMGeneration 的评估和测试设置的迁移。

评估字段分为 `val_evaluator` 和 `test_evaluator` ，并且评估字段不再支持 `interval` 和 `save_best` 参数。

- `interval` 移至 `train_cfg.val_interval`，请参阅[调度设置](./schedule.md)。
- `save_best` 移至 `default_hooks.checkpoint.save_best`。

<table class="docutils">
<thead>
  <tr>
    <th> 0.x 版本 </th>
    <th> 1.x 新版本 </th>
<tbody>
<tr>
<td valign="top">

```python
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(
            type='FID',
            num_images=50000,
            bgr2rgb=True,
            inception_args=dict(type='StyleGAN')),
        dict(type='IS', num_images=50000)
    ],
    best_metric=['fid', 'is'],
    sample_kwargs=dict(sample_model='ema'))
```

</td>

<td valign="top">

```python
val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='FID',
            prefix='FID-Full-50k',
            fake_nums=50000,
            inception_style='StyleGAN',
            sample_model='orig'),
        dict(
            type='IS',
            prefix='IS-50k',
            fake_nums=50000)])
# 设置最佳配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        less_keys=['FID-Full-50k/fid'],
        greater_keys=['IS-50k/is'],
        save_optimizer=True,
        save_best=['FID-Full-50k/fid', 'IS-50k/is'],
        rule=['less', 'greater']))
test_evaluator = val_evaluator
```

</td>

</tr>
</thead>
</table>

为了正确评估和测试模型，我们需要在 `val_cfg` 和 `test_cfg` 中设置特定的循环。

<table class="docutils">
<thead>
  <tr>
    <th> 0.x 版本中的静态模型 </th>
    <th> 1.x 版本中的静态模型 </th>
<tbody>
<tr>
<td valign="top">

```python
total_iters = 1000000

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
```

</td>

<td valign="top">

```python
train_cfg = dict(
    by_epoch=False,  # 使用基于迭代的训练
    max_iters=1000000,  # 最大训练迭代次数
    val_begin=1,
    val_interval=10000)  # 评价间隔
val_cfg = dict(type='MultiValLoop')  # 验证中的特定循环
test_cfg = dict(type='MultiTestLoop')  # 测试中的特定循环
```

</td>

</tr>
</thead>
</table>
