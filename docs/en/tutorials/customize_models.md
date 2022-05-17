# Tutorial 3: Customize Models

MMEditing supports multiple tasks, each of which has different settings. Fortunately, their customization is similar. Here, we use a super-resolution model, BasicVSR, as an example in this tutorial. You will be able to define your model based on your own needs after this tutorial.

We first need to create BasicVSR in `mmedit/models/restorers/basicvsr.py` .

```python
from ..registry import MODELS
from .basic_restorer import BasicRestorer

@MODELS.register_module()
class BasicVSR(BasicRestorer):

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))
```

## Model Argument

The values of these arguments are taken from the configuration file. Let's have a glance at the model part in the configuration file, you can find the complete config at `configs/restorers/basicvsr/basicvsr_reds4.py` .

```python
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRNet',
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)
```

We will now go through them one by one.

## generator

`generator` specifies the network architecture, which is called **backbone** in MMEditing. The definition of the backbone is straightforward, but there is one thing that needs our attention.

### Defining Backbone

Create a new file `mmedit/models/backbones/basicvsr_net.py` . The definition is standard. Please do make sure the line `@BACKBONES.register_module()` is added for all modules you would like to use.

```python
import torch.nn as nn

from ..builder import BACKBONES

@BACKBONES.register_module()
class BasicVSRNet(nn.Module):

    def __init__(self, mid_channels, num_blocks, spynet_pretrained):
        pass

    def forward(self, x):
        pass
```

### Importing Module

This is the part we need to be careful. We need to add the following line to `mmedit/models/backbones/__init__.py` to use the defined backbone.

```python
from .basicvsr_net import BasicVSRNet
```

## Specification in Configuration File

Given the above model, the specification in the configuration file is straightforward. We see that the argument `type` is just the name of the backbone, and other arguments correspond to that in the backbone.

```python
generator=dict(
    type='BasicVSRNet',
    mid_channels=64,
    num_blocks=30,
    spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
    'basicvsr/spynet_20210409-c6c1bd09.pth')
```

## pixel_loss

`pixel_loss` refers to the loss used in BasicVSR. The specification of the loss is similar to that of the backbone.

### Defining Loss

Let's use Charbonnier loss as an example. We first define the loss in `mmedit/models/losses/pixelwise_loss.py` . The decorator `masked_loss` enables the loss to be weighted and masked for each element. Again, do make sure that the line `@LOSSES.register_module()` is included.

```python
from ..registry import LOSSES
from .utils import masked_loss

@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

@LOSSES.register_module()
class CharbonnierLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
```

Similarly, we need to add the following line to `mmedit/models/losses/__init__.py` .

```python
from .pixelwise_loss import CharbonnierLoss
```

Then, the specification in the config follows naturally.

```python
pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean')
```

## train_cfg and test_cfg

`train_cfg` and `test_cfg` are just additional parameters you want to pass to the model. For example, in BasicVSR, a constant is passed to the model to fix a part of the network for a certain number of iterations.

```python
self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
```

## Model Functions

The model functions are used to control the training and test. In this tutorial, we will highlight a few important ones. For more details of the functions, you may refer to [here](https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/restorers/basic_restorer.py).

### train_step

This corresponds to the pipeline of each iteration, including forward and backward. In this example, the output and losses are computed. They are then used for backpropagation. More details of the forward process is discussed below.

```python
def train_step(self, data_batch, optimizer):
    outputs = self(**data_batch, test_mode=False)
    loss, log_vars = self.parse_losses(outputs.pop('losses'))

    # optimize
    optimizer['generator'].zero_grad()
    loss.backward()
    optimizer['generator'].step()

    outputs.update({'log_vars': log_vars})
    return outputs
```

### forward_train

This corresponds to the forward process. In this example, we will compute `output` given `lq` . Then `pixel_loss` is computed between `output` and `gt` . The computed loss will then be passed to a dictionary for further computations, including backpropagation. If you have any other losses, you should also include them here.

```python
def forward_train(self, lq, gt):
    losses = dict()
    output = self.generator(lq)
    loss_pix = self.pixel_loss(output, gt)
    losses['loss_pix'] = loss_pix
    outputs = dict(
        losses=losses,
        num_samples=len(gt.data),
        results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
    return outputs
```

### forward_test

This corresponds to the validation and test. For example, you need to specify how you perform evaluation (i.e. calculation of metrics) and how you save the outputs.

```python
def forward_test(self,
                lq,
                gt=None,
                meta=None,
                save_image=False,
                save_path=None,
                iteration=None):

        output = self.generator(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results
```
