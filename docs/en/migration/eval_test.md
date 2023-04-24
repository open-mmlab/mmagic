# Migration of Evaluation and Testing Settings

We update evaluation settings in MMagic 1.x. Important modifications are as following.

- The evaluation field is split to `val_evaluator` and `test_evaluator`. The `interval` is moved to `train_cfg.val_interval`.
- The metrics to evaluation are moved from `test_cfg` to `val_evaluator` and `test_evaluator`.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
train_cfg = None  # Training config
test_cfg = dict(  # Test config
    metrics=['PSNR'],  # Metrics used during testing
    crop_border=scale)  # Crop border during evaluation

evaluation = dict(  # The config to build the evaluation hook
    interval=5000,  # Evaluation interval
    save_image=True,  # Save images during evaluation
    gpu_collect=True)  # Use gpu collect
```

</td>

<td valign="top">

```python
val_evaluator = [
    dict(type='PSNR', crop_border=scale),  # The name of metrics to evaluate
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # Config of train loop type
val_cfg = dict(type='ValLoop')  # The name of validation loop type
test_cfg = dict(type='TestLoop')  # The name of test loop type
```

</td>

</tr>
</thead>
</table>

We have merged [MMGeneration 1.x](https://github.com/open-mmlab/mmgeneration/tree/1.x) into MMagic. Here is migration of Evaluation and Testing Settings about MMGeneration.

The evaluation field is splited to `val_evaluator` and `test_evaluator`. And it won't support `interval` and `save_best` arguments. The `interval` is moved to `train_cfg.val_interval`, see [the schedule settings](./schedule.md) and the `save_best` is moved to `default_hooks.checkpoint.save_best`.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
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
            sample_model='orig')
        dict(
            type='IS',
            prefix='IS-50k',
            fake_nums=50000)])
# set best config
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

To evaluate and test the model correctly, we need to set specific loop in `val_cfg` and `test_cfg`.

<table class="docutils">
<thead>
  <tr>
    <th> Static Model in 0.x Version </th>
    <th> Static Model in 1.x Version </th>
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
    by_epoch=False,  # use iteration based training
    max_iters=1000000,  # max training iteration
    val_begin=1,
    val_interval=10000)  # evaluation interval
val_cfg = dict(type='MultiValLoop')  # specific loop in validation
test_cfg = dict(type='MultiTestLoop')  # specific loop in testing
```

</td>

</tr>
</thead>
</table>
