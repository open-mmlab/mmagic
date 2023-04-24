# Migration of Model Settings

We update model settings in MMagic 1.x. Important modifications are as following.

- Remove `pretrained` fields.
- Add `train_cfg` and `test_cfg` fields in model settings.
- Add `data_preprocessor` fields. Normalization and color space transforms operations are moved from datasets transforms pipelines to data_preprocessor. We will introduce data_preprocessor later.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
model = dict(
    type='BasicRestorer',  # Name of the model
    generator=dict(  # Config of the generator
        type='EDSR',  # Type of the generator
        in_channels=3,  # Channel number of inputs
        out_channels=3,  # Channel number of outputs
        mid_channels=64,  # Channel number of intermediate features
        num_blocks=16,  # Block number in the trunk network
        upscale_factor=scale, # Upsampling factor
        res_scale=1,  # Used to scale the residual in residual block
        rgb_mean=(0.4488, 0.4371, 0.4040),  # Image mean in RGB orders
        rgb_std=(1.0, 1.0, 1.0)),  # Image std in RGB orders
    pretrained=None,
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))  # Config for pixel loss model training and testing settings
```

</td>

<td valign="top">

```python
model = dict(
    type='BaseEditModel',  # Name of the model
    generator=dict(  # Config of the generator
        type='EDSRNet',  # Type of the generator
        in_channels=3,  # Channel number of inputs
        out_channels=3,  # Channel number of outputs
        mid_channels=64,  # Channel number of intermediate features
        num_blocks=16,  # Block number in the trunk network
        upscale_factor=scale, # Upsampling factor
        res_scale=1,  # Used to scale the residual in residual block
        rgb_mean=(0.4488, 0.4371, 0.4040),  # Image mean in RGB orders
        rgb_std=(1.0, 1.0, 1.0)),  # Image std in RGB orders
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean')  # Config for pixel loss
    train_cfg=dict(),  # Config of training model.
    test_cfg=dict(),  # Config of testing model.
    data_preprocessor=dict(  # The Config to build data preprocessor
        type='DataPreprocessor', mean=[0., 0., 0.], std=[255., 255.,
                                                             255.]))
```

</td>

</tr>
</thead>
</table>

We refactor models in MMagic 1.x. Important modifications are as following.

- The `models` in MMagic 1.x is refactored to six parts: `archs`, `base_models`, `data_preprocessors`, `editors`, `diffusion_schedulers` and `losses`.
- Add `data_preprocessor` module in `models`. Normalization and color space transforms operations are moved from datasets transforms pipelines to data_preprocessor. The data out from the data pipeline is transformed by this module and then fed into the model.

More details of models are shown in [model guides](../howto/models.md).
