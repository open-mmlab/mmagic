# Migration of Schedule Settings

We update schedule settings in MMagic 1.x. Important modifications are as following.

- Now we use `optim_wrapper` field to specify all configuration about the optimization process. And the `optimizer` is a sub field of `optim_wrapper` now.
- The `lr_config` field is removed and we use new `param_scheduler` to replace it.
- The `total_iters` field is moved to `train_cfg` as `max_iters`, `val_cfg` and `test_cfg`, which configure the loop in training, validation and test.

<table class="docutils">
<thead>
  <tr>
    <th> Original </th>
    <th> New </th>
<tbody>
<tr>
<td valign="top">

```python
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
total_iters = 300000 # Total training iters
lr_config = dict( # Learning rate scheduler config used to register LrUpdater hook
    policy='Step', by_epoch=False, step=[200000], gamma=0.5)  # The policy of scheduler
```

</td>

<td valign="top">

```python
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4),
    )
)  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch.
param_scheduler = dict(  # Config of learning policy
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)  # The policy of scheduler
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300000, val_interval=5000)  # Config of train loop type
val_cfg = dict(type='ValLoop')  # The name of validation loop type
test_cfg = dict(type='TestLoop')  # The name of test loop type
```

</td>

</tr>
</thead>
</table>

> More details of schedule settings are shown in [MMEngine Documents](https://github.com/open-mmlab/mmengine/blob/main/docs/en/migration/param_scheduler.md).
