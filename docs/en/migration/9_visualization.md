# Migration of Visualization

In 0.x, MMGeneration use `MMGenVisualizationHook` and `VisualizeUnconditionalSamples` to visualization generating results in training process. In 1.x version, we unify the function of those hooks into `GenVisualizationHook`. Additionally, follow the design of MMEngine, we implement `GenVisualizer` and a group of `VisBackend` to draw and save the visualization results.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)
]
```

</td>

<td valign="top">

```python
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
vis_backends = [dict(type='GenVisBackend')]
visualizer = dict(type='GenVisualizer', vis_backends=vis_backends)
```

</td>

</tr>
</thead>
</table>

To learn more about the visualization function, please refers to [this tutorial](./user_guides/5_visualization.md).
