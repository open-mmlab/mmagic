# Migration of Runtime Settings

We update runtime settings in MMEdit 1.x. Important modifications are as following.

- The `checkpoint_config` is moved to `default_hooks.checkpoint` and the `log_config` is moved to `default_hooks.logger`. And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.
- The `resume_from` is removed. And we use `resume` to replace it.
  - If resume=True and load_from is not None, resume training from the checkpoint in load_from.
  - If resume=True and load_from is None, try to resume from the latest checkpoint in the work directory.
  - If resume=False and load_from is not None, only load the checkpoint, not resume training.
  - If resume=False and load_from is None, do not load nor resume.
- The `dist_params` field is a sub field of `env_cfg` now. And there are some new configurations in the `env_cfg`.
- The `workflow` related functionalities are removed.
- New field `visualizer`: The visualizer is a new design. We use a visualizer instance in the runner to handle results & log visualization and save to different backends,  like Local, TensorBoard and Wandb.
- New field `default_scope`: The start point to search module for all registries.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=5000,  # The save interval is 5000 iterations
    save_optimizer=True,  # Also save optimizers
    by_epoch=False)  # Count by iterations
log_config = dict(  # Config to register logger hook
    interval=100,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # The logger used to record the training process
        dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
    ])
visual_config = None  # Visual config, we do not use it.
# runtime settings
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
load_from = None # load models as a pre-trained model from a given path. This will not resume training
resume_from = None # Resume checkpoints from a given path, the training will be resumed from the iteration when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. Keep this unchanged when training current matting models
```

</td>

<td valign="top">

```python
default_hooks = dict(  # Used to build default hooks
    checkpoint=dict(  # Config to set the checkpoint hook
        type='CheckpointHook',
        interval=5000,  # The save interval is 5000 iterations
        save_optimizer=True,
        by_epoch=False,  # Count by iterations
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # Config to register logger hook
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
default_scope = 'mmedit' # Used to set registries location
env_cfg = dict(  # Parameters to setup distributed training, the port can also be set
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'  # The level of logging
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)  # Used to build log processor
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume = False  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
```

</td>

</tr>
</thead>
</table>

In 0.x version, MMGeneration use `total_iters` fields to control the total training iteration and use `DynamicIterBasedRunner` to handle the training process.
In 1.x version, we use [`Runner`](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/runner.md) and [`Loops`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py) provided by `MMEngine` and use `train_cfg.max_iters` field to control the total training iteration and use `train_cfg.val_interval` to control the evaluation interval.

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
val_cfg = dict(type='GenValLoop')  # specific loop in validation
test_cfg = dict(type='GenTestLoop')  # specific loop in testing
```

</td>

</tr>
</thead>
</table>

In 0.x version, MMGeneration use `total_iters` fields to control the total training iteration and use `DynamicIterBasedRunner` to handle the training process.
In 1.x version, we use [`Runner`](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/runner.md) and [`Loops`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py) provided by `MMEngine` and use `train_cfg.max_iters` field to control the total training iteration and use `train_cfg.val_interval` to control the evaluation interval.

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
val_cfg = dict(type='GenValLoop')  # specific loop in validation
test_cfg = dict(type='GenTestLoop')  # specific loop in testing
```

</td>

</tr>
</thead>
</table>

Changes in **`checkpoint_config`** and **`log_config`**:

The `checkpoint_config` are moved to `default_hooks.checkpoint` and the `log_config` are moved to `default_hooks.logger`.
And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.

```python
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type='GenIterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        less_keys=['FID-Full-50k/fid'],
        greater_keys=['IS-50k/is'],
        save_optimizer=True))
```

In addition, we splited the original logger to logger and visualizer. The logger is used to record
information and the visualizer is used to show the logger in different backends, like terminal, TensorBoard
and Wandb.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
```

</td>
<tr>
<td>New</td>
<td>

```python
default_hooks = dict(
    ...
    logger=dict(type='LoggerHook', interval=100),
)

```

</td>
</tr>
</table>

Changes in **`load_from`** and **`resume_from`**:

- The `resume_from` is removed. And we use `resume` and `load_from` to replace it.
  - If `resume=True` and `load_from` is not None, resume training from the checkpoint in `load_from`.
  - If `resume=True` and `load_from` is None, try to resume from the latest checkpoint in the work directory.
  - If `resume=False` and `load_from` is not None, only load the checkpoint, not resume training.
  - If `resume=False` and `load_from` is None, do not load nor resume.

Changes in **`dist_params`**: The `dist_params` field is a sub field of `env_cfg` now. And there are some new
configurations in the `env_cfg`.

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'))
```

Changes in **`workflow`**: `workflow` related functionalities are removed.

New field **`default_scope`**: The start point to search module for all registries. The `default_scope` in MMGeneration is `mmgen`. See [the registry tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md) for more details.
