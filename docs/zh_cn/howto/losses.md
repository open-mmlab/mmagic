# 如何设计自己的损失函数

`losses` 在 `MMagic` 中注册为 `LOSSES`。
在 MMagic 中设计自己的损失函数，步骤和在 MMagic 中自定义任何其他模型类似。
本节主要具体介绍了如何在 MMagic 中实现自定义的损失函数。
本教程建议您在实现自定义的损失函数时，应该遵循本教程相同的设计，这样在我们的框架中使用您新定义的损失函数，就不需要额外的工作。

本指南包括：

- [设计你自己的损失函数](#如何设计自己的损失函数)
  - [支持损失函数介绍](#支持的损失函数介绍)
  - [设计一个新的损失函数](#设计一个新的损失函数)
    - [MSELoss 的一个例子](#MSELoss-的一个例子)
    - [DiscShiftLoss 的一个例子](#DiscShiftLoss-的一个例子)
    - [GANWithCustomizedLoss 的一个例子](#GANWithCustomizedLoss-的一个例子)
  - [可用损失函数](#可用损失函数)
    - [常规损失函数](#常规损失函数)
    - [损失函数组件](#损失函数组件)

## 支持的损失函数介绍

为了方便使用，您可以直接使用我们为具体算法设置的默认损失计算过程，如lsgan、biggan、styleganv2等。
以`stylegan2`为例，我们使用R1梯度惩罚和生成器路径长度正则化作为可配置损失，用户可以调整相关参数，如 `r1_loss_weight` 和 `g_reg_weight`。

```python
# stylegan2_base.py
loss_config =dict(
     r1_loss_weight=10。 / 2. * d_reg_interval,
     r1_interval=d_reg_interval,
     norm_mode='HWC',
     g_reg_interval=g_reg_interval,
     g_reg_weight=2。 * g_reg_interval,
     pl_batch_shrink=2)

model=dict(
     type='StyleGAN2',
     xxx,
     loss_config=loss_config)
```

## 设计一个新的损失函数

### MSELoss 的一个例子

一般来说，要实现一个损失模块，我们会编写一个函数实现，然后用类实现包装它。 以MSELoss为例：

```python
@masked_loss
def mse_loss(pred，target)：
     return F.mse_loss（pred，target，reduction='none'）

@LOSSES.register_module()
Class MSELoss(nn.Module)：

     def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
         # 代码可以在``mmagic/models/losses/pixelwise_loss.py``中找到

     def forward(self, pred, target, weight=None, **kwargs):
         # 代码可以在``mmagic/models/losses/pixelwise_loss.py``中找到
```

根据这个损失函数的定义，我们现在可以简单地通过在配置文件中定义它来使用：

```python
pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean')
```

请注意，上面的`pixel_loss`必须在模型中定义。 详情请参考[自定义模型](./models.md)。 与自定义模型类似，为了使用您自己实现的损失函数，您需要在编写后在`mmagic/models/losses/__init__.py`中导入该损失函数。

### DiscShiftLoss 的一个例子

一般来说，要实现一个损失模块，我们会编写一个函数实现，然后用类实现包装它。
但是，在 MMagic 中，我们提供了另一个统一的接口 data_info 供用户定义输入参数和数据项之间的映射。

```python
@weighted_loss
def disc_shift_loss(pred)：
     return pred**2

@MODULES.register_module()
Class DiscShiftLoss(nn.Module)：

     def __init__(self, loss_weight=1.0, data_info=None):
         super(DiscShiftLoss，self).__init__()
         # 代码可以在``mmagic/models/losses/disc_auxiliary_loss.py``中找到

     def forward(self, *args, **kwargs):
         # 代码可以在``mmagic/models/losses/disc_auxiliary_loss.py``中找到
```

这种损失模块设计的目标是允许在生成模型(`MODELS`)中自动使用它，而无需其他复杂代码来定义数据和关键字参数之间的映射。 因此，与 OpenMMLab 中的其他框架不同，我们的损失模块包含一个特殊的关键字 data_info，它是一个定义输入参数与生成模型数据之间映射的字典。 以`DiscShiftLoss`为例，用户在编写配置文件时，可能会用到这个loss，如下：

```python
dict(type='DiscShiftLoss',
    loss_weight=0.001 * 0.5,
    data_info=dict(pred='disc_pred_real'))
```

`data_info` 中的信息告诉模块使用 `disc_pred_real` 数据作为 `pred` 参数的输入张量。 一旦 `data_info` 不为 `None`，我们的损失模块将自动构建计算图。

```python
@MODULES.register_module()
class DiscShiftLoss(nn.Module):

    def __init__(self, loss_weight=1.0, data_info=None):
        super(DiscShiftLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info

    def forward(self, *args, **kwargs):
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(dict(weight=self.loss_weight))
            return disc_shift_loss(**kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return disc_shift_loss(*args, weight=self.loss_weight, **kwargs)

    @staticmethod
    def loss_name():
        return 'loss_disc_shift'

```

如这部分代码所示，一旦用户设置了“data_info”，损失模块将收到一个包含所有必要数据和模块的字典，该字典由训练过程中的“MODELS”提供。 如果此字典作为非关键字参数给出，则应将其作为第一个参数提供。 如果您使用关键字参数，请将其命名为 `outputs_dict`。

### GANWithCustomizedLoss 的一个例子

为了构建计算图，生成模型必须提供包含各种数据的字典。 仔细观察任何生成模型，你会发现我们将各种特征和模块收集到字典中。 我们在这里提供了一个自定义的`GANWithCustomizedLoss`来展示这个过程。

```python
class GANWithCustomizedLoss(BaseModel):

    def __init__(self, gan_loss, disc_auxiliary_loss, gen_auxiliary_loss,
                 *args, **kwargs):
        # ...
        if gan_loss is not None:
            self.gan_loss = MODULES.build(gan_loss)
        else:
            self.gan_loss = None

        if disc_auxiliary_loss:
            self.disc_auxiliary_losses = MODULES.build(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = MODULES.build(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        # ...

        # get data dict to compute losses for disc
        data_dict_ = dict(
            iteration=curr_iter,
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # ...

    def _get_disc_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake'] = self.gan_loss(
            outputs_dict['disc_pred_fake'], target_is_real=False, is_disc=True)
        losses_dict['loss_disc_real'] = self.gan_loss(
            outputs_dict['disc_pred_real'], target_is_real=True, is_disc=True)

        # disc auxiliary loss
        if self.with_disc_auxiliary_loss:
            for loss_module in self.disc_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self.parse_losses(losses_dict)

        return loss, log_var

```

在这里，`_get_disc_loss` 将帮助自动组合各种损失函数。

因此，只要用户设计相同规则的损失模块，就可以在生成模型的训练中插入任何一种损失，无需对模型代码进行其他修改。 您只需要在配置文件中定义 `data_info` 即可。

## 可用损失函数

我们在配置中列出了可用的损失示例，如下所示。

### 常规损失函数

<table class="docutils">
<thead>
  <tr>
    <th>Method</th>
    <th>class</th>
    <th>Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>vanilla gan loss</td>
    <td>mmagic.models.GANLoss</td>
<td>

```python
# dic gan
loss_gan=dict(
    type='GANLoss',
    gan_type='vanilla',
    loss_weight=0.001,
)

```

</td>

</tr>
  <tr>
    <td>lsgan loss</td>
    <td>mmagic.models.GANLoss</td>
<td>
</td>

</tr>
  <tr>
    <td>wgan loss</td>
    <td>mmagic.models.GANLoss</td>
    <td>

```python
# deepfillv1
loss_gan=dict(
    type='GANLoss',
    gan_type='wgan',
    loss_weight=0.0001,
)
```

</td>

</tr>
  <tr>
    <td>hinge loss</td>
    <td>mmagic.models.GANLoss</td>
    <td>

```python
# deepfillv2
loss_gan=dict(
    type='GANLoss',
    gan_type='hinge',
    loss_weight=0.1,
)
```

</td>

</tr>
  <tr>
    <td>smgan loss</td>
    <td>mmagic.models.GANLoss</td>
<td>

```python
# aot-gan
loss_gan=dict(
    type='GANLoss',
    gan_type='smgan',
    loss_weight=0.01,
)
```

</td>

</tr>
  <tr>
    <td>gradient penalty</td>
    <td>mmagic.models.GradientPenaltyLoss</td>
    <td>

```python
# deepfillv1
loss_gp=dict(type='GradientPenaltyLoss', loss_weight=10.)
```

</td>

</tr>
  <tr>
    <td>discriminator shift loss</td>
    <td>mmagic.models.DiscShiftLoss</td>
    <td>

```python
# deepfillv1
loss_disc_shift=dict(type='DiscShiftLoss', loss_weight=0.001)

```

</td>

</tr>
  <tr>
    <td>clip loss</td>
    <td>mmagic.models.CLIPLoss</td>
    <td></td>

</tr>
  <tr>
    <td>L1 composition loss</td>
    <td>mmagic.models.L1CompositionLoss</td>
    <td></td>

</tr>
  <tr>
    <td>MSE composition loss</td>
    <td>mmagic.models.MSECompositionLoss</td>
    <td></td>

</tr>
  <tr>
    <td>charbonnier composition loss</td>
    <td>mmagic.models.CharbonnierCompLoss</td>
    <td>

```python
# dim
loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5)
```

</td>

</tr>
  <tr>
    <td>face id Loss</td>
    <td>mmagic.models.FaceIdLoss</td>
    <td></td>

</tr>
  <tr>
    <td>light cnn feature loss</td>
    <td>mmagic.models.LightCNNFeatureLoss</td>
    <td>

```python
# dic gan
feature_loss=dict(
    type='LightCNNFeatureLoss',
    pretrained=pretrained_light_cnn,
    loss_weight=0.1,
    criterion='l1')
```

</td>

</tr>
  <tr>
    <td>gradient loss</td>
    <td>mmagic.models.GradientLoss</td>
    <td></td>

</tr>
  <tr>
    <td>l1 Loss</td>
    <td>mmagic.models.L1Loss</td>
    <td>

```python
# dic gan
pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean')
```

</td>

</tr>
  <tr>
    <td>mse loss</td>
    <td>mmagic.models.MSELoss</td>
    <td>

```python
# dic gan
align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean')
```

</td>

</tr>
  <tr>
    <td>charbonnier loss</td>
    <td>mmagic.models.CharbonnierLoss</td>
    <td>

```python
# dim
loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5)
```

</td>

</tr>
  <tr>
    <td>masked total variation loss</td>
    <td>mmagic.models.MaskedTVLoss</td>
    <td>

```python
# partial conv
loss_tv=dict(
    type='MaskedTVLoss',
    loss_weight=0.1
)

```

</td>

</tr>
  <tr>
    <td>perceptual loss</td>
    <td>mmagic.models.PerceptualLoss</td>
    <td>

```python
# real_basicvsr
perceptual_loss=dict(
    type='PerceptualLoss',
    layer_weights={
        '2': 0.1,
        '7': 0.1,
        '16': 1.0,
        '25': 1.0,
        '34': 1.0,
    },
    vgg_type='vgg19',
    perceptual_weight=1.0,
    style_weight=0,
    norm_img=False)

```

</td>

</tr>
  <tr>
    <td>transferal perceptual loss</td>
    <td>mmagic.models.TransferalPerceptualLoss</td>
    <td>

```python
# ttsr
transferal_perceptual_loss=dict(
    type='TransferalPerceptualLoss',
    loss_weight=1e-2,
    use_attention=False,
    criterion='mse')

```

</td>
  </tr>
</tbody>
</table>

### 损失函数组件

对于“GANWithCustomizedLoss”，我们提供了几个组件来构建自定义损失。

| Method                               | class                                       |
| ------------------------------------ | ------------------------------------------- |
| clip loss component                  | mmagic.models.CLIPLossComps                 |
| discriminator shift loss component   | mmagic.models. DiscShiftLossComps           |
| gradient penalty loss component      | mmagic.models. GradientPenaltyLossComps     |
| r1 gradient penalty component        | mmagic.models. R1GradientPenaltyComps       |
| face Id loss component               | mmagic.models. FaceIdLossComps              |
| gan loss component                   | mmagic.models. GANLossComps                 |
| generator path regularizer component | mmagic.models.GeneratorPathRegularizerComps |
