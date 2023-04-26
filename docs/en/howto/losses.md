# How to design your own loss functions

`losses` are registered as `LOSSES` in `MMagic`.
Customizing losses is similar to customizing any other model.
This section is mainly for clarifying the design of loss modules in MMagic.
Importantly, when writing your own loss modules, you should follow the same design,
so that the new loss module can be adopted in our framework without extra effort.

This guides includes:

- [How to design your own loss functions](#how-to-design-your-own-loss-functions)
  - [Introduction to supported losses](#introduction-to-supported-losses)
  - [Design a new loss function](#design-a-new-loss-function)
    - [An example of MSELoss](#an-example-of-mseloss)
    - [An example of DiscShiftLoss](#an-example-of-discshiftloss)
    - [An example of GANWithCustomizedLoss](#an-example-of-ganwithcustomizedloss)
  - [Available losses](#available-losses)
    - [regular losses](#regular-losses)
    - [losses components](#losses-components)

## Introduction to supported losses

For convenient usage, you can directly use default loss calculation process we set for concrete algorithms like lsgan, biggan, styleganv2 etc.
Take `stylegan2` as an example, we use R1 gradient penalty and generator path length regularization as configurable losses, and users can adjust
related arguments like `r1_loss_weight` and `g_reg_weight`.

```python
# stylegan2_base.py
loss_config = dict(
    r1_loss_weight=10. / 2. * d_reg_interval,
    r1_interval=d_reg_interval,
    norm_mode='HWC',
    g_reg_interval=g_reg_interval,
    g_reg_weight=2. * g_reg_interval,
    pl_batch_shrink=2)

model = dict(
    type='StyleGAN2',
    xxx,
    loss_config=loss_config)
```

## Design a new loss function

### An example of MSELoss

In general, to implement a loss module, we will write a function implementation and then wrap it with a class implementation. Take the MSELoss as an example:

```python
@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@LOSSES.register_module()
class MSELoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        # codes can be found in ``mmagic/models/losses/pixelwise_loss.py``

    def forward(self, pred, target, weight=None, **kwargs):
        # codes can be found in ``mmagic/models/losses/pixelwise_loss.py``
```

Given the definition of the loss, we can now use the loss by simply defining it in the configuration file:

```python
pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean')
```

Note that `pixel_loss` above must be defined in the model. Please refer to `customize_models` for more details. Similar to model customization, in order to use your customized loss, you need to import the loss in `mmagic/models/losses/__init__.py` after writing it.

### An example of DiscShiftLoss

In general, to implement a loss module, we will write a function implementation and then wrap it with a class implementation.
However, in `MMagic`, we provide another unified interface `data_info` for users to define the mapping between the input argument and data items.

```python
@weighted_loss
def disc_shift_loss(pred):
    return pred**2

@MODULES.register_module()
class DiscShiftLoss(nn.Module):

    def __init__(self, loss_weight=1.0, data_info=None):
        super(DiscShiftLoss, self).__init__()
        # codes can be found in ``mmagic/models/losses/disc_auxiliary_loss.py``

    def forward(self, *args, **kwargs):
        # codes can be found in ``mmagic/models/losses/disc_auxiliary_loss.py``
```

The goal of this design for loss modules is to allow for using it automatically in the generative models (`MODELS`), without other complex codes to define the mapping between data and keyword arguments. Thus, different from other frameworks in `OpenMMLab`, our loss modules contain a special keyword, `data_info`, which is a dictionary defining the mapping between the input arguments and data from the generative models. Taking the `DiscShiftLoss` as an example, when writing the config file, users may use this loss as follows:

```python
dict(type='DiscShiftLoss',
    loss_weight=0.001 * 0.5,
    data_info=dict(pred='disc_pred_real')
```

The information in `data_info` tells the module to use the `disc_pred_real` data as the input tensor for `pred` arguments. Once the `data_info` is not `None`, our loss module will automatically build up the computational graph.

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

As shown in this part of codes, once users set the `data_info`, the loss module will receive a dictionary containing all of the necessary data and modules, which is provided by the `MODELS` in the training procedure. If this dictionary is given as a non-keyword argument, it should be offered as the first argument. If you are using a keyword argument, please name it as `outputs_dict`.

### An example of GANWithCustomizedLoss

To build the computational graph, the generative models have to provide a dictionary containing all kinds of data. Having a close look at any generative model, you will find that we collect all kinds of features and modules into a dictionary. We provide a customized `GANWithCustomizedLoss` here to show the process.

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

Here, the `_get_disc_loss` will help to combine all kinds of losses automatically.

Therefore, as long as users design the loss module with the same rules, any kind of loss can be inserted in the training of generative models,
without other modifications in the code of models. What you only need to do is just defining the `data_info` in the config files.

## Available losses

We list available losses with examples in configs as follows.

### regular losses

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

### losses components

For `GANWithCustomizedLoss`, we provide several components to build customized loss.

| Method                               | class                                       |
| ------------------------------------ | ------------------------------------------- |
| clip loss component                  | mmagic.models.CLIPLossComps                 |
| discriminator shift loss component   | mmagic.models. DiscShiftLossComps           |
| gradient penalty loss component      | mmagic.models. GradientPenaltyLossComps     |
| r1 gradient penalty component        | mmagic.models. R1GradientPenaltyComps       |
| face Id loss component               | mmagic.models. FaceIdLossComps              |
| gan loss component                   | mmagic.models. GANLossComps                 |
| generator path regularizer component | mmagic.models.GeneratorPathRegularizerComps |
