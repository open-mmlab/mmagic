# Migration of Visualization

In 0.x, MMEditing use `VisualizationHook` to visualize results in training process. In 1.x version, we unify the function of those hooks into `BasicVisualizationHook` / `VisualizationHook`. Additionally, follow the design of MMEngine, we implement `ConcatImageVisualizer` / `Visualizer` and a group of `VisBackend` to draw and save the visualization results.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
visual_config = dict(
    type='VisualizationHook',
    output_dir='visual',
    interval=1000,
    res_name_list=['gt_img', 'masked_img', 'fake_res', 'fake_img'],
)
```

</td>

<td valign="top">

```python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
```

</td>

</tr>
</thead>
</table>

To learn more about the visualization function, please refers to [this tutorial](../user_guides/visualization.md).
