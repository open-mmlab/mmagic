# Visualization

Using `visualizer` in config file can save visual results when training or testing. You can follow [MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/visualization.md) to learn the usage of visualization.
MMEditing supports `Local` visual backend and `Tensorboard ` visual backend now.

## LocalVisBackend

Here are configs using `LocalVisBackend` to save visual results as an example.
The output of models formatted to `DataSample` will be passed to `visualizer`. You can set the `img_keys` fields of `visualizer` to determine which values of images to visualize.

```python
vis_backends = [dict(type='LocalVisBackend')] # The type of visual backend
visualizer = dict( # Config to build visualizer
    type='ConcatImageVisualizer', # Type of visualizer
    vis_backends=vis_backends, # Type of visual backend
    fn_key='gt_path', # The key used to determine file name for saving image
    img_keys=['gt_img', 'input', 'pred_img'], # The key in img_keys will be visualized
    bgr2rgb=True) # Whether convert channels of image
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Config of visualization hook
```

Then visual results will be saved in `work_dirs` where configured in the config file.

## TensorboardVisBackend

Here are configs using `TensorboardVisBackend` to save visual results as an example.

```python
vis_backends = [dict(type='TensorboardVisBackend')] # The type of visual backend
visualizer = dict( # Config to build visualizer
    type='ConcatImageVisualizer', # Type of visualizer
    vis_backends=vis_backends, # Type of visual backend
    fn_key='gt_path', # The key used to determine file name for saving image
    img_keys=['gt_img', 'input', 'pred_img'], # The key in img_keys will be visualized
    bgr2rgb=True) # Whether convert channels of image
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)] # Config of visualization hook
```

You can start TensorBoard server and see visual results by visiting TensorBoard in the browser.
