# 运行设置的迁移

我们更新了 MMagic 1.x 中的运行设置，重要修改如下：

- `checkpoint_config` 被移动到 `default_hooks.checkpoint`，`log_config` 被移动到 `default_hooks.logger`。 我们将许多 hooks 设置从脚本代码移动到运行配置的 `default_hooks` 字段中。
- `resume_from` 被移除，使用 `resume` 替代它。
  - 如果 resume=True 并且 load_from 不是 None, 则从load_from中的检查点恢复训练。
  - 如果 resume=True 且 load_from 为 None，则尝试从工作目录中的最新检查点恢复。
  - 如果 resume=False 且 load_from 不为None，则仅加载检查点，不恢复训练。
  - 如果 resume=False 且 load_from 为 None，则不加载也不恢复。
- `dist_params` 字段现在是 `env_cfg` 的一个子字段。 并且在 `env_cfg` 还有一些新的配置。
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
checkpoint_config = dict(  # 设置检查点 hook 的配置, 参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py 完成的
    interval=5000,  # 保存间隔为 5000 次迭代
    save_optimizer=True,  # 也保存优化器
    by_epoch=False)  # 通过 iterations 计数
log_config = dict(  # 注册日志 hook 的配置
    interval=100,  # 打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # logger 用来记录训练过程
        dict(type='TensorboardLoggerHook'),  # 也支持 Tensorboard logger
    ])
visual_config = None  # 可视化配置，我们不使用它。
# runtime settings
dist_params = dict(backend='nccl')  # 设置分布式训练的参数，还可以设置端口
log_level = 'INFO'  # 日志等级
load_from = None # 从指定路径加载预训练模型，这不会恢复训练
resume_from = None # 从给定路径恢复检查点，训练将从保存检查点的epoch开始恢复
workflow = [('train', 1)]  # Runner 的工作流程. [('train', 1)] 意味着只有一个工作流，并且名为“train”的工作流执行一次。 在训练当前的抠图模型时，请保持此项不变
```

</td>

<td valign="top">

```python
default_hooks = dict(  # 用来创建默认 hooks
    checkpoint=dict(  # 设置 checkpoint hook 的配置
        type='CheckpointHook',
        interval=5000,  # 保存间隔为5000次迭代
        save_optimizer=True,
        by_epoch=False,  # 通过 iterations 计数
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),  # 注册 logger hook 的配置
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
default_scope = 'mmedit' # 用来设置注册位置
env_cfg = dict(  # 设置分布式训练的参数，还可以设置端口
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'  # 日志等级
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)  # 用来创建日志处理器
load_from = None  # 从指定路径加载预训练模型，这不会恢复训练
resume = False  # 从给定路径恢复检查点，训练将从保存检查点的epoch开始恢复
```

</td>

</tr>
</thead>
</table>
