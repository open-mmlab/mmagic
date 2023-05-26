# 优化器的迁移

我们已经将[MMGeneration 1.x](https://github.com/open-mmlab/mmgeneration/tree/1.x)合并至MMagic。以下是针对MMGeneration中优化器的迁移事项。

在0.x版中，MMGeneration使用PyTorch自带的优化器，其只提供了通用参数优化，而在1.x版中，我们则使用了MMEngine提供的`OptimizerWrapper`。

对比PyTorch自带的`Optimizer`，`OptimizerWrapper`可以支持如下功能：

- `OptimizerWrapper.update_params`在一个单一的函数中就实现了`zero_grad`，`backward`和`step`
- 支持梯度自动累积
- 提供一个名为`OptimizerWrapper.optim_context`的上下文管理器来封装前向进程，`optim_context`会根据当前更新迭代数目来自动调用`torch.no_sync`，在AMP(Auto Mixed Precision)训练中，`autocast`也会在`optim_context`中被调用。

对GAN模型，生成器和鉴别器采用不同的优化器和训练策略。要使GAN模型的`train_step`函数签名和其它模型的保持一致，我们使用从`OptimizerWrapper`继承下来的`OptimWrapperDict`来封装生成器和鉴别器的优化器，为了便于该流程的自动化MMagic实现了`MultiOptimWrapperContructor`构造器。如你想训练GAN模型，那么应该在你的配置中指定该构造器。

如下是0.x版和1.x版的配置对比

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
</tbody>

</table>

> 注意，在1.x版中，MMGeneration使用`OptimWrapper`来实现梯度累加，这就会导致在0.x版和1.x版之间，`discriminator_steps`配置（用于在多次更新鉴别器之后更新一次生成器的训练技巧）与梯度累加均出现不一致问题。

- 在0.x版中，我们在配置里使用`disc_steps`，`gen_steps`和`batch_accumulation_steps` 。`disc_steps`和`batch_accumulation_steps`会根据`train_step`的调用次数来进行统计（亦即dataloader中数据的读取次数）。因此鉴别器的一段连续性更新次数为`disc_steps // batch_accumulation_steps`。且对于生成器，`gen_steps`是生成器实际的一段连续性更新次数
- 但在1.x版中，我们在配置里则使用了`discriminator_steps`，`generator_steps` 和`accumulative_counts`。`discriminator_steps`和`generator_steps`指的是自身在更新其它模型之前的一段连续性的更新次数
  以BigGAN-128配置为例。

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

# 连续性更新鉴别器`disc_steps // batch_accumulation_steps = 8 // 8 = 1`次
# 连续性更新生成器`gen_steps = 1`次
# 生成器与鉴别器在每次更新之前执行`batch_accumulation_steps = 8`次梯度累加
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
    # 连续性更新鉴别器`discriminator_steps = 1`次
    # 连续性更新生成器`generator_steps = 1`次
    generator_steps=1,
    discriminator_steps=1)

optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        # 生成器在每次更新之前执行`accumulative_counts = 8`次梯度累加
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6)),
    discriminator=dict(
        # 鉴别器在每次更新之前执行`accumulative_counts = 8`次梯度累加
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6)))
```

</td>

</tr>
</tbody>

</table>
