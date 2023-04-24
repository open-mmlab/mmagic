# Migration of Optimizers

We have merged [MMGeneration 1.x](https://github.com/open-mmlab/mmgeneration/tree/1.x) into MMagic. Here is migration of Optimizers about MMGeneration.

In version 0.x, MMGeneration uses PyTorch's native Optimizer, which only provides general parameter optimization.
In version 1.x, we use `OptimizerWrapper` provided by MMEngine.

Compared to PyTorch's `Optimizer`, `OptimizerWrapper` supports the following features:

- `OptimizerWrapper.update_params` implement `zero_grad`, `backward` and `step` in a single function.
- Support gradient accumulation automatically.
- Provide a context manager named `OptimizerWrapper.optim_context` to warp the forward process. `optim_context` can automatically call `torch.no_sync` according to current number of updating iteration. In AMP (auto mixed precision) training, `autocast` is called in `optim_context` as well.

For GAN models, generator and discriminator use different optimizer and training schedule.
To ensure that the GAN model's function signature of `train_step` is consistent with other models, we use `OptimWrapperDict`, inherited from `OptimizerWrapper`, to wrap the optimizer of the generator and discriminator.
To automate this process MMagic implement `MultiOptimWrapperContructor`.
And you should specify this constructor in your config is you want to train GAN model.

The config for the 0.x and 1.x versions are shown below:

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6),
    discriminator=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6))
```

</td>

<td valign="top">

```python
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999), eps=1e-6)),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6)))
```

</td>

</tr>
</thead>
</table>

> Note that, in the 1.x, MMGeneration uses `OptimWrapper` to realize gradient accumulation. This make the config of `discriminator_steps` (training trick for updating the generator once after multiple updates of the discriminator) and gradient accumulation different between 0.x and 1.x version.

- In 0.x version,  we use `disc_steps`, `gen_steps` and `batch_accumulation_steps` in configs. `disc_steps` and `batch_accumulation_steps` are counted by the number of calls of `train_step` (is also the number of data reads from the dataloader). Therefore the number of consecutive updates of the discriminator is `disc_steps // batch_accumulation_steps`. And for generators, `gen_steps` is the number of times the generator actually updates continuously.
- In 1.x version, we use `discriminator_steps`, `generator_steps` and `accumulative_counts` in configs. `discriminator_steps` and `generator_steps` are the number of consecutive updates to itself before updating other modules.

Take config of BigGAN-128 as example.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
model = dict(
    type='BasiccGAN',
    generator=dict(
        type='BigGANGenerator',
        output_scale=128,
        noise_size=120,
        num_classes=1000,
        base_channels=96,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        split_noise=True,
        auto_sync_bn=False),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=128,
        num_classes=1000,
        base_channels=96,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))

# continuous update discriminator for `disc_steps // batch_accumulation_steps = 8 // 8 = 1` times
# continuous update generator for `gen_steps = 1` times
# generators and discriminators perform `batch_accumulation_steps = 8` times gradient accumulations before each update
train_cfg = dict(
    disc_steps=8, gen_steps=1, batch_accumulation_steps=8, use_ema=True)
```

</td>

<td valign="top">

```python
model = dict(
    type='BigGAN',
    num_classes=1000,
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(
        type='BigGANGenerator',
        output_scale=128,
        noise_size=120,
        num_classes=1000,
        base_channels=96,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        split_noise=True,
        auto_sync_bn=False),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=128,
        num_classes=1000,
        base_channels=96,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True),
    # continuous update discriminator for `discriminator_steps = 1` times
    # continuous update generator for `generator_steps = 1` times
    generator_steps=1,
    discriminator_steps=1)

optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        # generator perform `accumulative_counts = 8` times gradient accumulations before each update
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6)),
    discriminator=dict(
        # discriminator perform `accumulative_counts = 8` times gradient accumulations before each update
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6)))
```

</td>

</tr>
</thead>
</table>
