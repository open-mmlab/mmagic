# 可视化的迁移

在0.x版中，MMEditing使用`VisualizationHook`来对训练过程中生成的结果进行可视化，在1.x版中，我们将该功能整合到`BasicVisualizationHook` / `VisualizationHook`中，而且遵循MMEngine的设计，我们实现了`ConcatImageVisualizer` / `Visualizer`和一系列`VisBackend`来绘制和保存可视化结果。

<table class="docutils">
<thead>
  <tr>
    <th> 0.x版 </th>
    <th> 1.x版 </th>
  </tr>
</thead>
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
</tbody>
</table>

要了解更多关于可视化的功能，请参阅[这个教程](../user_guides/visualization.md)。
