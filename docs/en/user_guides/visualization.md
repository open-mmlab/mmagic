# Visualization

Using `visualizer` in config file can save visual results when training or testing.
MMEditing supports `Loacl` visual backend and `Tensorboard ` visual backend now.

## LocalVisBackend

Here are configs using `LocalVisBackend` to save visual results as an example.

```python
vis_backends = [dict(type='LocalVisBackend')] # The type of visual backend
visualizer = dict( # Config to build visualizer
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Config of visualization hook
```

## TensorboardVisBackend

Here are configs using `TensorboardVisBackend` to save visual results as an example.

```python
vis_backends = [dict(type='TensorboardVisBackend')] # The type of visual backend
visualizer = dict( # Config to build visualizer
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Config of visualization hook
```
