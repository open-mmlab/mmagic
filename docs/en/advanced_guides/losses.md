# Tutorial 4: Customize Losses

`losses` are registered as `LOSSES` in `MMEditing`. Customizing losses is similar to customizing any other model. This section is mainly for clarifying the design of loss modules in our repo. Importantly, when writing your own loss modules, you should follow the same design, so that the new loss module can be adopted in our framework without extra effort.

## Design of loss modules

In general, to implement a loss module, we will write a function implementation and then wrap it with a class implementation. Take the MSELoss as an example:

```python
@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@LOSSES.register_module()
class MSELoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        # codes can be found in ``mmedit/models/losses/pixelwise_loss.py``

    def forward(self, pred, target, weight=None, **kwargs):
        # codes can be found in ``mmedit/models/losses/pixelwise_loss.py``
```

Given the definition of the loss, we can now use the loss by simply defining it in the configuration file:

```python
pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean')
```

Note that `pixel_loss` above must be defined in the model. Please refer to `customize_models` for more details. Similar to model customization, in order to use your customized loss, you need to import the loss in `mmedit/models/losses/__init__.py` after writing it.
