# 运行设置的迁移

我们更新了 MMagic 1.x 中的运行设置，重要修改如下：

- `checkpoint_config` 被移动到 `default_hooks.checkpoint`，`log_config` 被移动到 `default_hooks.logger`。 我们将许多 hooks 设置从脚本代码移动到运行配置的 `default_hooks` 字段中。
- `resume_from` 被移除，使用 `resume` 替代它。
  - 如果 resume=True 并且 load_from 不是 None, 则从load_from中的检查点恢复训练。
  - 如果 resume=True 且 load_from 为 None，则尝试从工作目录中的最新检查点恢复。
  - 如果 resume=False 且 load_from 不为None，则仅加载检查点，不恢复训练。
  - 如果 resume=False 且 load_from 为 None，则不加载也不恢复。
- `dist_params` 字段现在是一个子字段 `env_cfg` 。 并且在 `env_cfg` 还有一些新的配置。
- `workflow` 相关功能已被删除。
- 新字段 `visualizer`: 可视化工具是一个新设计。在 runner 中使用可视化器实例来处理结果和日志可视化并保存到不同的后端，例如 Local、TensorBoard 和 Wandb。
- 新字段 `default_scope`: 所有注册器搜索 module 的起点。


<table class="docutils">
<thead>
  <tr>
    <th> 原始配置 </th>
    <th> 新的配置 </th>
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
